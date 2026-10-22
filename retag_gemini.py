import pandas as pd
import json
import re
import argparse
from difflib import SequenceMatcher

def find_best_substring_match(needle, haystack):
    """
    다듬어진 문장(haystack) 안에서 원본 Node 텍스트(needle)와 
    가장 유사한 부분의 시작, 끝 인덱스를 찾습니다.
    """
    if not needle or not haystack:
        return None, None, 0

    # 1. 정확히 일치하는 경우 우선 반환
    if needle in haystack:
        start = haystack.find(needle)
        return start, start + len(needle), 1.0

    # 2. 정확한 일치가 없으면 유사도 검색 (SequenceMatcher)
    # 문장이 길 경우 속도 최적화를 위해 sliding window 방식 등을 고려할 수 있으나,
    # 여기서는 정확도를 위해 difflib의 매처를 사용합니다.
    matcher = SequenceMatcher(None, needle, haystack)
    match = matcher.find_longest_match(0, len(needle), 0, len(haystack))
    
    if match.size == 0:
        return None, None, 0
    
    # 찾은 부분이 너무 짧거나(예: 1글자), 의미가 없을 경우 필터링할 수도 있음
    # 여기서는 가장 긴 매치 부분을 기준으로 확장하거나 그대로 사용
    
    # (고도화: 단순 longest match가 아니라, 해당 match 주변으로 
    # 원본 needle 길이만큼의 영역을 잡고 점수를 계산하는 것이 더 정확할 수 있음)
    # 이 스크립트에서는 '가장 길게 일치하는 부분'을 기준으로 삼습니다.
    
    return match.b, match.b + match.size, match.size / len(needle)

def extract_tags_from_raw(rr_text):
    """
    paragraph_rr_raw 텍스트에서 태그 정보와 내부 텍스트를 추출합니다.
    Format: <role: attribute> content </role>
    Return: list of dicts [{'tag_type': 'source', 'attr': ': direct cause...', 'content': 'A-d-a highly'}]
    """
    # Regex 패턴: <tag (속성)> 내용 </tag>
    # 예: <source: direct cause of X> Y </source>
    # group 1: tag_type (e.g., source)
    # group 2: attribute (e.g., : direct cause of X) - 있을수도 없을수도 있음
    # group 3: content (e.g., Y)
    # group 4: closing tag (e.g., source)
    
    pattern = re.compile(r"<(\w+)(:[^>]+)?>(.*?)</(\w+)>")
    matches = []
    
    for m in pattern.finditer(rr_text):
        tag_type_start = m.group(1)
        attr = m.group(2) if m.group(2) else ""
        content = m.group(3).strip()
        tag_type_end = m.group(4)
        
        if tag_type_start == tag_type_end:
            matches.append({
                'tag_type': tag_type_start,
                'attr': attr,
                'content': content
            })
            
    return matches

def process_row(row):
    """
    각 행에 대해 태그 이전을 수행합니다.
    """
    rr_raw = row.get('paragraph_rr_raw', '')
    plain_smooth = row.get('paragraph_plain_smoothed_smooth', '')
    
    if pd.isna(rr_raw) or pd.isna(plain_smooth):
        return None, None

    # 1. Raw 데이터에서 태그 정보 추출
    tags_info = extract_tags_from_raw(rr_raw)
    
    # 2. Smoothed 문장에서 각 태그의 새로운 위치 찾기
    replacements = []
    
    for tag in tags_info:
        raw_content = tag['content']
        
        # 다듬어진 문장에서 원본 내용과 가장 유사한 위치 찾기
        start, end, score = find_best_substring_match(raw_content, plain_smooth)
        
        if start is not None and score > 0.3: # 유사도 threshold (너무 낮은 매칭 제외)
            replacements.append({
                'start': start,
                'end': end,
                'tag_type': tag['tag_type'],
                'attr': tag['attr'],
                'content': plain_smooth[start:end] # 매칭된 실제 텍스트
            })
        else:
            # 매칭 실패 시 경고(선택사항) 혹은 건너뜀
            pass

    # 3. 위치 순서대로 정렬 (뒤에서부터 처리하거나, 재구성 방식 사용)
    # 겹치는 구간이 있을 경우를 대비해 start 기준으로 정렬
    replacements.sort(key=lambda x: x['start'])
    
    # 4. 문자열 재구성
    # 주의: 인덱스가 꼬이지 않도록 오프셋을 계산하거나 리스트로 조립
    
    res_role = []
    res_rr = []
    last_idx = 0
    
    for rep in replacements:
        # 태그 시작 전까지의 텍스트 추가
        pre_text = plain_smooth[last_idx:rep['start']]
        res_role.append(pre_text)
        res_rr.append(pre_text)
        
        # 태그가 입혀진 텍스트 추가
        # role: <source> content </source>
        # rr: <source: attr> content </source>
        
        match_text = rep['content'] # 다듬어진 텍스트
        
        role_tag = f"<{rep['tag_type']}>{match_text}</{rep['tag_type']}>"
        rr_tag = f"<{rep['tag_type']}{rep['attr']}>{match_text}</{rep['tag_type']}>"
        
        res_role.append(role_tag)
        res_rr.append(rr_tag)
        
        last_idx = rep['end']
        
    # 마지막 남은 텍스트 추가
    res_role.append(plain_smooth[last_idx:])
    res_rr.append(plain_smooth[last_idx:])
    
    return "".join(res_role), "".join(res_rr)

def main(input_file, output_file):
    print(f"Reading from {input_file}...")
    df = pd.read_csv(input_file)
    
    new_role_col = []
    new_rr_col = []
    
    print("Processing rows...")
    for idx, row in df.iterrows():
        try:
            role_smooth, rr_smooth = process_row(row)
            new_role_col.append(role_smooth)
            new_rr_col.append(rr_smooth)
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            new_role_col.append(row.get('paragraph_plain_smoothed_smooth')) # 에러 시 원문 유지
            new_rr_col.append(row.get('paragraph_plain_smoothed_smooth'))

    # 5. 새로운 컬럼 추가
    df['paragraph_role_smoothed_smooth'] = new_role_col
    df['paragraph_rr_smoothed_smooth'] = new_rr_col
    
    print(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer tags from raw text to smoothed text.")
    parser.add_argument("--input", "-i", type=str, default= 'path_paragraphs_final3.csv', help="Input CSV file path")
    parser.add_argument("--output", "-o", type=str, default = 'path_paragraphs_final3_retag_gem.csv', help="Output CSV file path")
    
    args = parser.parse_args()
    
    main(args.input, args.output)