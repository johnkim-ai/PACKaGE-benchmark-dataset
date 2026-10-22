#!/usr/bin/env python3
"""
Paragraph Tagging Script
이 스크립트는 다듬어진 문장(paragraph_plain_smoothed_smooth)에 
원본과 동일한 방식의 role tagging을 추가합니다.

사용법:
python add_paragraph_tags.py input.csv output.csv
"""

import pandas as pd
import json
import re
import argparse
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional, Set


def find_node_in_text(node_text: str, smoothed_text: str, used_positions: Set[Tuple[int, int]]) -> Optional[Tuple[int, int, str]]:
    """
    노드 텍스트를 smoothed 문장에서 찾습니다.
    
    Args:
        node_text: 찾을 노드 텍스트
        smoothed_text: 검색할 문장
        used_positions: 이미 사용된 위치들 (겹침 방지)
    
    Returns:
        (start_idx, end_idx, matched_text) 또는 None
    """
    
    # 1. 정확한 매칭 시도
    start_idx = 0
    while True:
        idx = smoothed_text.find(node_text, start_idx)
        if idx == -1:
            break
        
        end_idx = idx + len(node_text)
        # 이미 사용된 위치와 겹치지 않는지 확인
        is_overlapping = any(
            not (end_idx <= used_start or idx >= used_end)
            for used_start, used_end in used_positions
        )
        
        if not is_overlapping:
            return (idx, end_idx, node_text)
        
        start_idx = idx + 1
    
    # 2. 부분 매칭 시도 (단어 단위로 점진적으로 줄여가며)
    words = node_text.split()
    
    for length in range(len(words), 0, -1):
        for i in range(len(words) - length + 1):
            partial = ' '.join(words[i:i + length])
            
            start_idx = 0
            while True:
                idx = smoothed_text.find(partial, start_idx)
                if idx == -1:
                    break
                
                end_idx = idx + len(partial)
                # 겹침 확인
                is_overlapping = any(
                    not (end_idx <= used_start or idx >= used_end)
                    for used_start, used_end in used_positions
                )
                
                if not is_overlapping:
                    return (idx, end_idx, partial)
                
                start_idx = idx + 1
    
    # 3. 유사도 기반 매칭 (노드의 첫 단어로 시작하는 위치 찾기)
    if words:
        first_word = words[0]
        # 단어 경계를 고려한 패턴
        pattern = r'\b' + re.escape(first_word) + r'\b'
        
        best_match = None
        best_score = 0
        
        for match in re.finditer(pattern, smoothed_text, re.IGNORECASE):
            start = match.start()
            
            # 이 위치부터 적절한 길이의 텍스트 추출
            # 원본 노드 길이의 0.5배 ~ 2배 범위에서 탐색
            min_len = max(5, len(node_text) // 2)
            max_len = min(len(smoothed_text) - start, len(node_text) * 2)
            
            for end in range(start + max_len, start + min_len - 1, -1):
                candidate = smoothed_text[start:end]
                
                # 겹침 확인
                is_overlapping = any(
                    not (end <= used_start or start >= used_end)
                    for used_start, used_end in used_positions
                )
                
                if is_overlapping:
                    continue
                
                # 유사도 계산
                score = SequenceMatcher(None, node_text.lower(), candidate.lower()).ratio()
                
                # 단어 경계에서 끝나는지 확인 (가산점)
                if end < len(smoothed_text) and smoothed_text[end] in ' .,;:!?\n':
                    score += 0.1
                
                if score > best_score and score > 0.5:  # 최소 유사도 임계값
                    best_score = score
                    best_match = (start, end, candidate)
        
        if best_match:
            return best_match
    
    return None


def process_row(row: pd.Series) -> pd.Series:
    """
    각 행을 처리하여 paragraph_role_smoothed_smooth와 
    paragraph_rr_smoothed_smooth를 생성합니다.
    """
    smoothed_text = str(row['paragraph_plain_smoothed_smooth'])
    graph_json_str = row['graph_json']
    
    try:
        graph_data = json.loads(graph_json_str)
    except json.JSONDecodeError as e:
        print(f"Warning: JSON parsing error in row - {e}")
        return pd.Series({
            'paragraph_role_smoothed_smooth': smoothed_text,
            'paragraph_rr_smoothed_smooth': smoothed_text
        })
    
    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', [])
    
    # 노드 ID를 키로 하는 딕셔너리 생성
    node_dict = {node['id']: node for node in nodes}
    
    # 각 노드의 역할 파악 (source, leaf)
    sources = set()
    targets = set()
    edge_map = {}  # source -> [targets]
    reverse_edge_map = {}  # target -> [sources]
    
    for edge in edges:
        src = edge['source']
        tgt = edge['target']
        sources.add(src)
        targets.add(tgt)
        
        if src not in edge_map:
            edge_map[src] = []
        edge_map[src].append(tgt)
        
        if tgt not in reverse_edge_map:
            reverse_edge_map[tgt] = []
        reverse_edge_map[tgt].append(src)
    
    # 노드를 문장에서 찾기
    node_spans = []
    used_positions = set()
    
    # 노드를 원본 문장에서의 순서대로 정렬 (대략적)
    # 실제로는 smoothed 문장에서 찾은 위치로 정렬할 것
    for node in nodes:
        node_id = node['id']
        
        span_info = find_node_in_text(node_id, smoothed_text, used_positions)
        
        if span_info:
            start, end, matched_text = span_info
            
            # 역할 결정
            if node_id in sources and node_id not in targets:
                role = 'source'
            elif node_id in targets and node_id not in sources:
                role = 'leaf'
            elif node_id in sources and node_id in targets:
                # intermediate node - source로 취급
                role = 'source'
            else:
                # isolated node - 태그 안 함
                continue
            
            node_spans.append({
                'node_id': node_id,
                'start': start,
                'end': end,
                'text': matched_text,
                'role': role
            })
            
            used_positions.add((start, end))
    
    # 위치 순으로 정렬
    node_spans.sort(key=lambda x: x['start'])
    
    # paragraph_role_smoothed_smooth 생성
    role_tagged = generate_role_tagged(smoothed_text, node_spans)
    
    # paragraph_rr_smoothed_smooth 생성
    rr_tagged = generate_rr_tagged(smoothed_text, node_spans, edge_map, reverse_edge_map)
    
    return pd.Series({
        'paragraph_role_smoothed_smooth': role_tagged,
        'paragraph_rr_smoothed_smooth': rr_tagged
    })


def generate_role_tagged(text: str, node_spans: List[Dict]) -> str:
    """
    role 태그만 추가 (<source>, <leaf>)
    """
    result = text
    offset = 0
    
    for span in node_spans:
        role = span['role']
        start = span['start'] + offset
        end = span['end'] + offset
        
        if role == 'source':
            open_tag = '<source>'
            close_tag = '</source>'
        elif role == 'leaf':
            open_tag = '<leaf>'
            close_tag = '</leaf>'
        else:
            continue
        
        result = result[:start] + open_tag + result[start:end] + close_tag + result[end:]
        offset += len(open_tag) + len(close_tag)
    
    return result


def generate_rr_tagged(text: str, node_spans: List[Dict], 
                       edge_map: Dict[str, List[str]], 
                       reverse_edge_map: Dict[str, List[str]]) -> str:
    """
    role과 relationship 태그 추가
    (<source: direct cause of X>, <leaf: direct effect of Y>)
    
    중요: relation 텍스트 안에는 다른 태그가 들어가면 안 됨!
    """
    result = text
    offset = 0
    
    for span in node_spans:
        role = span['role']
        node_id = span['node_id']
        start = span['start'] + offset
        end = span['end'] + offset
        
        if role == 'source':
            # 이 노드가 cause가 되는 target들
            targets = edge_map.get(node_id, [])
            if targets:
                # 여러 target이 있을 경우 쉼표로 구분
                relation_text = 'direct cause of ' + ', '.join(targets)
                open_tag = f'<source: {relation_text}>'
            else:
                open_tag = '<source>'
            close_tag = '</source>'
            
        elif role == 'leaf':
            # 이 노드가 effect가 되는 source들
            sources = reverse_edge_map.get(node_id, [])
            if sources:
                relation_text = 'direct effect of ' + ', '.join(sources)
                open_tag = f'<leaf: {relation_text}>'
            else:
                open_tag = '<leaf>'
            close_tag = '</leaf>'
            
        else:
            continue
        
        result = result[:start] + open_tag + result[start:end] + close_tag + result[end:]
        offset += len(open_tag) + len(close_tag)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Add role tags to smoothed paragraphs based on graph structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python add_paragraph_tags.py input.csv output.csv
  python add_paragraph_tags.py data/paragraphs.csv data/paragraphs_tagged.csv

입력 CSV는 다음 컬럼을 포함해야 합니다:
  - paragraph_plain_smoothed_smooth: 다듬어진 문장
  - graph_json: 노드와 엣지 정보가 담긴 JSON
        """
    )
    
    parser.add_argument('input_csv', default = 'path_paragraphs_final3.csv', help='입력 CSV 파일 경로')
    parser.add_argument('output_csv', default = 'path_paragraphs_final3_claude.csv', help='출력 CSV 파일 경로')
    parser.add_argument('--encoding', default='utf-8', 
                       help='CSV 파일 인코딩 (기본값: utf-8)')
    
    args = parser.parse_args()
    
    print(f"입력 파일 읽는 중: {args.input_csv}")
    try:
        df = pd.read_csv(args.input_csv, encoding=args.encoding)
    except FileNotFoundError:
        print(f"Error: 파일을 찾을 수 없습니다 - {args.input_csv}")
        return
    except Exception as e:
        print(f"Error: CSV 파일 읽기 실패 - {e}")
        return
    
    # 필수 컬럼 확인
    required_columns = ['paragraph_plain_smoothed_smooth', 'graph_json']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: 필수 컬럼이 없습니다 - {missing_columns}")
        print(f"현재 컬럼: {list(df.columns)}")
        return
    
    print(f"총 {len(df)} 행 처리 중...")
    
    # 새 컬럼 생성
    result_df = df.apply(process_row, axis=1)
    df['paragraph_role_smoothed_smooth'] = result_df['paragraph_role_smoothed_smooth']
    df['paragraph_rr_smoothed_smooth'] = result_df['paragraph_rr_smoothed_smooth']
    
    # 저장
    print(f"결과 저장 중: {args.output_csv}")
    df.to_csv(args.output_csv, index=False, encoding=args.encoding)
    
    print(f"완료! {len(df)} 행이 처리되었습니다.")
    print(f"새로 추가된 컬럼:")
    print(f"  - paragraph_role_smoothed_smooth")
    print(f"  - paragraph_rr_smoothed_smooth")


if __name__ == '__main__':
    main()