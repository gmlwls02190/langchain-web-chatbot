from tkinter import Image
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field


# 1. 워크플로우 단계 정의
class workflowStep:
    GREETING = "GREETING"
    PROCESSING = "PROCESSING"

# 2. 그래프 상태 정의 : 그래프의 모든 노드가 읽고 쓰는 공유 데이터 구조, 각 노드는 이 상태를 입력으로 받고, 바꾸고 싶은 필드만 반환하면 기존 상태에서 자동으로 업데이트 한다.
class GraphState(BaseModel): # 클래스명에서 각단어마다 대문자로 시작하는 방식을 파스칼 방식
    name: str = Field(default="", description="사용자 이름")    # 입력값
    greeting: str = Field(default="", description="생성된 인사말")  # GRRETING 노드의 결과
    processing_message: str = Field(default="", description="처리된 최종 메시지")   # PROCESSING 결과

# 3. 첫번째 노드 함수 : 인사말 생성 함수
def generate_greeting(state: GraphState) -> Dict[str, Any]:
    name = state.name or "홍길동"
    greeting = f"안녕하세요, {name}님!"
    print(f"[generate_greeting] 인사말 생성: {greeting}")
    return {"greeting": greeting}

# 4. 두번째 노드 함수 : 인사말을 처리하고 최종 메시지 작성
def processing_message(state: GraphState) -> Dict[str, Any]:
    greeting = state.greeting
    processing_message = f"{greeting} LangGraph World에 오신것을 환영합니다!"
    print(f"[processing_message] 최종 메시지: {processing_message}")
    return {"processing_message": processing_message}

# 5. 그래프 생성 
def create_hello_graph():
    workflow = StateGraph(GraphState)
    # 노드 추가
    workflow.add_node(workflowStep.GREETING, generate_greeting)
    workflow.add_node(workflowStep.PROCESSING, processing_message)
    # 시작점 설정
    workflow.add_edge(START, workflowStep.GREETING)
    # 엣지 추가(노드 간 연결)
    workflow.add_edge(workflowStep.GREETING, workflowStep.PROCESSING)
    # 끝점 설정
    workflow.add_edge(workflowStep.PROCESSING, END)
    # 그래프 컴파일
    app = workflow.compile()
    return app

# 실행
def main():
    print("=== Hello 랭그래프 ===\n")
    app = create_hello_graph()

    initial_state = GraphState(name="승귤", greeting="", processing_message="")
    print("초기 상태:", initial_state.model_dump())
    print("\n--- 그래프 실행 시작 ---")

    # 그래프 실행
    final_state = app.invoke(initial_state)

    print("--- 그래프 실행 종료 ---\n")
    print("최종 상태:", final_state)
    print(f"\n결과 메시지: {final_state['processing_message']}")
    # ⑥ ASCII로 그래프 출력
    result = app.get_graph().draw_ascii()
    print(result)

    # ⑦ Mermaid로 그래프 이미지 생성
    result = app.get_graph().draw_mermaid_png()

    with open("./hello_langgraph.png", "wb") as f:
        f.write(result)


if __name__ == "__main__":
    main()
