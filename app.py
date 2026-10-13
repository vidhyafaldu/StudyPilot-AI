
import streamlit as st
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

API_URL = "http://127.0.0.1:8000/v1/systemone"
MODEL_NAME = "Qwen/Qwen3.5-2B-Base"

st.set_page_config(
    page_title="StudyPilot AI",
    page_icon="📚",
    layout="centered"
)


# ---------- LOCAL AI ----------

@st.cache_resource
def load_local_ai():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="cpu"
    )

    model.eval()
    return tokenizer, model


def generate_local_answer(topic, action):
    tokenizer, model = load_local_ai()

    tasks = {
        "explain_topic": (
            "Explain the topic with a definition, "
            "simple points and an example."
        ),
        "create_study_plan": (
            "Create a practical study timetable "
            "with study sessions and revision."
        ),
        "generate_quiz": (
            "Create five multiple-choice questions "
            "with four options and correct answers."
        ),
        "review_progress": (
            "Give a study progress checklist and "
            "practical improvement suggestions."
        )
    }

    instruction = tasks.get(action, tasks["explain_topic"])

    prompt = (
        "You are StudyPilot AI, a helpful study tutor.\n"
        f"Topic: {topic}\n"
        f"Task: {instruction}\n"
        "Use simple English, clear headings and examples.\n"
        "Answer:\n"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()


# ---------- STUDYPILOT AI UI ----------

st.title("📚 StudyPilot AI")
st.write("Your smart study assistant")
st.caption("Decision engine: Strands Decider 2B")

topic = st.text_input(
    "What do you want to study?",
    placeholder="Example: Python Loops, DBMS, Artificial Intelligence"
)

goal = st.selectbox(
    "What help do you need?",
    [
        "Let AI decide",
        "Help me understand this topic",
        "Create a study plan",
        "Generate a quiz",
        "Review my study progress"
    ]
)

action_map = {
    "Help me understand this topic": "explain_topic",
    "Create a study plan": "create_study_plan",
    "Generate a quiz": "generate_quiz",
    "Review my study progress": "review_progress"
}

criteria = {
    "explain_topic": "Explain the requested topic simply.",
    "create_study_plan": "Create a practical study timetable.",
    "generate_quiz": "Generate practice questions on the topic.",
    "review_progress": "Review the student's study progress."
}


if st.button("Generate Study Answer", type="primary"):

    if not topic.strip():
        st.warning("Please enter a topic first.")

    else:
        try:
            # Ask Decider only when the user wants AI to choose.
            if goal == "Let AI decide":

                payload = {
                    "state": (
                        f"Student needs help with: {topic}. "
                        "Choose the most useful study action."
                    ),
                    "model": "strands-decider-latest",
                    "questions": {
                        "next_action": {
                            "type": "choice",
                            "instructions": (
                                "Choose the most useful action "
                                "for this student's topic."
                            ),
                            "criteria": criteria
                        }
                    }
                }

                with st.spinner("Decider is choosing an action..."):
                    response = requests.post(
                        API_URL,
                        json=payload,
                        timeout=180
                    )
                    response.raise_for_status()
                    data = response.json()

                selected_action = (
                    data["answers"]["next_action"]["choice"]
                )

                confidence = data["answers"]["next_action"].get(
                    "confidence"
                )

                st.success("Decider selected an action!")

                if confidence is not None:
                    st.write(f"Confidence: {confidence:.2%}")

            else:
                # Respect the action selected by the user.
                selected_action = action_map[goal]

            st.subheader("Selected study action")
            st.code(selected_action)

            with st.spinner(
                "Local AI is preparing your answer. "
                "The first run may take a few minutes..."
            ):
                answer = generate_local_answer(
                    topic,
                    selected_action
                )

            st.subheader("📚 Your Study Answer")

            if answer:
                st.markdown(answer)
            else:
                st.warning(
                    "The model returned an empty answer. Try again."
                )

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to Decider. "
                "Make sure the server is running at "
                "http://127.0.0.1:8000."
            )

        except requests.exceptions.Timeout:
            st.error(
                "The Decider request timed out. "
                "Check that the server is running and try again."
            )

        except Exception as error:
            st.error("StudyPilot AI could not generate the answer.")
            st.code(f"{type(error).__name__}: {error}")
