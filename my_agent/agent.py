import asyncio
from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

model = LiteLlm(
    model="ollama/llama3",
    api_base="http://localhost:11434"
)

root_agent = Agent(
    model=model,
    name="root_agent",
    description="A helpful assistant for user questions.",
    instruction="Answer user questions to the best of your knowledge. Use previous conversation context if available."
)

async def main():
    session_service = InMemorySessionService()
    session_id = "session1"
    await session_service.create_session(
        app_name="my_app",
        user_id="user",
        session_id=session_id
    )

    runner = Runner(
        agent=root_agent,
        app_name="my_app",
        session_service=session_service
    )

    print("Start asking questions! (type 'exit' to quit)")

    while True:
        query = input("Ask a question: ")
        if query.lower() == "exit":
            break

        content = Content(role="user", parts=[Part(text=query)])
        final_text = None

        async for event in runner.run_async(
            user_id="user",
            session_id=session_id,
            new_message=content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_text = event.content.parts[0].text

        print("Response:", final_text)

if __name__ == "__main__":
    asyncio.run(main())