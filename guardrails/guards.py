# Import Guard and Validator
from guardrails import Guard
from guardrails.hub import (
    RestrictToTopic,
    WebSanitization,
    ValidURL,
    LogicCheck
)

class GitBuddyGuardrails():
    # Setup Guards
    restrict_to_topic_guard = Guard().use(
        RestrictToTopic(
            valid_topics=["git", "github", "tortoisegit", "gitlab", "branching", "github copilot", "commits", "git commands", "repositories", "version control", "CI/CD", "GitHub Actions", "GitHub Desktop"],
            disable_classifier=True,
            disable_llm=False,
            llm_callable="gpt-4o",
            on_fail="exception"
        )
    )

    web_sanitization_guard = Guard.use(
        WebSanitization()
    )

    valid_url_guard = Guard().use(
        ValidURL(on_fail="exception")
    )

    logic_check_guard = Guard().use(
        LogicCheck()
    )