from utils import (
    get_search_query,
    parse_urls,
    remove_specific_string_from_list,
    remove_specific_element_from_list,
    reduce_chat_history_tokens,
    reduce_similar_docs_tokens,
)


def test_get_search_query():
    sources = ["test\\query1.", "example\\query2."]
    expected = ["query1", "query2"]
    assert get_search_query(sources) == expected


def test_parse_urls():
    search_results = "Check out this link: https://example.com"
    expected = ["https://example.com"]
    assert parse_urls(search_results) == expected


def test_remove_specific_string_from_list():
    nested_list = [["string1_remove", "string2_remove"], ["string3_remove"]]
    string_to_remove = "_remove"
    expected = [["string1", "string2"], ["string3"]]
    assert remove_specific_string_from_list(nested_list, string_to_remove) == expected


def test_remove_specific_element_from_list():
    nested_list = [["element1", "remove"], ["element2", "element3", "remove"]]
    element_to_remove = "remove"
    expected = [["element1"], ["element2", "element3"]]
    assert remove_specific_element_from_list(nested_list, element_to_remove) == expected


def test_reduce_chat_history_tokens():
    chat_history_dict = {
        "chat_history": """Human: What can you tell me about GitHub milestones?
                        AI: GitHub milestones are a way to track progress and organize work in a repository. They can be used to define specific goals or targets for a project. Milestones can be created and edited within a repository, and they can be applied to issues, pull requests, or discussions. They provide a way to categorize and filter work based on a specific milestone.
                        Human: What about issues? how do those impact milestones
                        AI: Issues in GitHub are used to track and manage tasks, bugs, feature requests, and other work items in a project. They allow teams to organize and prioritize their work. 

                        Milestones, on the other hand, are used to group related issues together and track progress towards a specific goal or deadline. They provide a way to organize and categorize issues based on their importance or timeframe. 

                        When working with issues, you can assign them to a milestone to indicate that they are part of a larger goal or project. This helps in tracking the progress of the milestone and provides a high-level overview of the work that needs to be done. 

                        For example, if you have a milestone for a new product release, you can assign related issues to that milestone to track their progress towards the release. This allows you to easily see the status of all the issues that are part of the release and ensure that everything is on track.

                        To learn more about issues and milestones in GitHub, you can refer to the following sources:
                        - [GitHub Docs: About issues](https://docs.github.com/en/issues/tracking-your-work-with-issues/about-issues)
                        - [TechRepublic: How to create a GitHub issue](https://www.techrepublic.com/article/how-to-create-github-issue/)
                        - [Rewind: Best practices for using GitHub issues](https://rewind.com/blog/best-practices-for-using-github-issues/)
                        - [Dev.to: What are GitHub issues?](https://dev.to/devshetty/what-are-github-issues-4c3p)
                        Human: and how is tortoisegit useful?
                        AI: TortoiseGit is a Windows-based graphical user interface (GUI) for Git. It provides an easy-to-use interface for performing common Git operations such as commit, push, pull, and merge. TortoiseGit integrates seamlessly with Windows Explorer, allowing you to interact with Git repositories directly from the file explorer.

                        Here are some ways TortoiseGit can be useful:

                        1. User-friendly interface: TortoiseGit provides a user-friendly interface that makes it easier for beginners to understand and use Git. You can perform Git operations with just a few clicks, without needing to remember complex command-line commands.

                        2. Integration with Windows Explorer: TortoiseGit integrates with Windows Explorer, allowing you to perform Git operations directly from the file explorer. You can see the status of files, commit changes, and perform other Git operations by right-clicking on files or folders.

                        3. Visual diff and merge: TortoiseGit provides a visual diff and merge tool, making it easier to review and merge changes between different versions of files. You can easily see the differences between files and resolve conflicts during the merge process.

                        4. Commit history visualization: TortoiseGit provides a graphical representation of the commit history, allowing you to easily navigate and understand the evolution of a project. You can view the commit log, browse individual commits, and compare different versions of files.

                        5. Integration with remote repositories: TortoiseGit supports integration with remote repositories hosted on platforms like GitHub, Bitbucket, and GitLab. You can clone, push, and pull repositories directly from TortoiseGit, making it easy to collaborate with others.

                        Overall, TortoiseGit provides a user-friendly and intuitive interface for working with Git, making it a valuable tool for beginners and experienced users alike.

                        Additional Sources:
                        - [TortoiseGit Official Website](https://tortoisegit.org/)
                        - [TortoiseGit Tutorial](https://www.softwaretestinghelp.com/tortoisegit-tutorial/)
                        - [TortoiseGit Tutorial on MarketSplash](https://marketsplash.com/tutorials/git/tortoise-git/)
                        Interaction4
                        Interaction5
                        Interaction6"""
    }
    token_limit = 10  # Assuming each interaction is 2 tokens
    reduced_dict = reduce_chat_history_tokens(chat_history_dict, token_limit)
    assert (
        """Human: What can you tell me about GitHub milestones?
              AI: GitHub milestones are a way to track progress and organize work in a repository. They can be used to define specific goals or targets for a project. Milestones can be created and edited within a repository, and they can be applied to issues, pull requests, or discussions. They provide a way to categorize and filter work based on a specific milestone."""
        not in reduced_dict["chat_history"]
    )
    assert (
        """          Human: What about issues? how do those impact milestones
                            AI: Issues in GitHub are used to track and manage tasks, bugs, feature requests, and other work items in a project. They allow teams to organize and prioritize their work.

                            Milestones, on the other hand, are used to group related issues together and track progress towards a specific goal or deadline. They provide a way to organize and categorize issues based on their importance or timeframe.

                            When working with issues, you can assign them to a milestone to indicate that they are part of a larger goal or project. This helps in tracking the progress of the milestone and provides a high-level overview of the work that needs to be done.

                            For example, if you have a milestone for a new product release, you can assign related issues to that milestone to track their progress towards the release. This allows you to easily see the status of all the issues that are part of the release and ensure that everything is on track.

                            To learn more about issues and milestones in GitHub, you can refer to the following sources:
                            - [GitHub Docs: About issues](https://docs.github.com/en/issues/tracking-your-work-with-issues/about-issues)
                            - [TechRepublic: How to create a GitHub issue](https://www.techrepublic.com/article/how-to-create-github-issue/)
                            - [Rewind: Best practices for using GitHub issues](https://rewind.com/blog/best-practices-for-using-github-issues/)
                            - [Dev.to: What are GitHub issues?](https://dev.to/devshetty/what-are-github-issues-4c3p)
    """
        in reduced_dict["chat_history"]
    )
    assert "Interaction6" in reduced_dict["chat_history"]


def test_reduce_similar_docs_tokens():
    similar_docs = ["doc1", "doc2", "doc3", "doc4"]
    token_limit = 10  # Set a token limit
    reduced_docs = reduce_similar_docs_tokens(similar_docs, token_limit)
    assert len(reduced_docs) <= len(similar_docs)  # Check if documents were removed
    assert "doc1" not in reduced_docs  # Assuming doc1 was removed due to token limit
