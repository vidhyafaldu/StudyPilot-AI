
import json

name = input("Enter your name: ")
goal = input("Enter your AI goal: ")

profile = {
    "name": name,
    "goal": goal,
    "topics": [
        "Python",
        "AI Agents",
        "Machine Learning"
    ]
}

print("\nYour AI Learning Profile:")
print(json.dumps(profile, indent=2))
