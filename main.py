import sys
from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question



load_dotenv()

def run_pipeline(source :str, language :str = "english") -> dict:
    print("\n🚀 Starting  Recallr...\n")

    # Step 1: Process input
    chunks = process_input(source)

    # Step 2: Transcribe
    transcript = transcribe_all(chunks,language)

    # Step 3: Summarize & generate metadata
    title = generate_title(transcript)
    summary = summarize(transcript)
    action_item = extract_action_items(transcript)
    decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)
    
    # Step 4: Build RAG chain
    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_item,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


def cli():
    try:
        source = input("📂 Enter YouTube URL or local file path: ").strip()
        if not source:
            print("❌ No source provided. Exiting.")
            sys.exit(1)
        
        language = input("🌐 Language (english/hinglish) [default: english]: ").strip() or "english"

        result = run_pipeline(source, language)

        print("\n" + "=" * 60)
        print(f"📌 Title: {result['title']}")
        print(f"\n📋 Summary:\n{result['summary']}")
        print(f"\n✅ Action Items:\n{result['action_items']}")
        print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
        print(f"\n❓ Open Questions:\n{result['open_questions']}")
        print("=" * 60)

        # Interactive RAG chat
        print("\n💬 Chat with your Local video / Youtube video (type 'exit' or 'quit'or 'q' to quit)\n")
        rag_chain = result["rag_chain"]

        while True:
            question = input("You: ").strip()
            if question.lower() in {"exit", "quit", "q"}:
                print("👋 Goodbye!")
                break
            if not question:
                continue
            answer = ask_question(rag_chain, question)
            print(f"\n🤖 Recallr: {answer}\n")

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user. Exiting.")
        sys.exit(0)
    except Exception as e:
        print(f"\n Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    cli()