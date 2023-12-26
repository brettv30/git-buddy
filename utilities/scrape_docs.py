from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from bs4 import BeautifulSoup as Soup
import re


def remove_extra_whitespace(my_str):
    # Remove useless page content from GitHub docs
    string_to_remove1 = "GitHub DocsSkip to main contentGitHub DocsVersion: Free, Pro, & TeamSearch GitHub DocsGet started/Exploring integrationsHomeGet startedQuickstartCreate an accountHello WorldSet up GitGitHub flowBe socialCommunicating on GitHubGitHub glossaryGit cheatsheetLearning resourcesOnboardingGetting started with your GitHub accountGetting started with GitHub TeamGetting started with GitHub Enterprise CloudLearning about GitHubGitHub’s plansGitHub language supportTypes of GitHub accountsAccess permissionsGitHub Advanced SecurityChanges to GitHub plansUsing GitHubConnecting to GitHubFeature previewSupported browsersGitHub MobileAllow network accessConnectivity problemsAccessibilityManage theme settingsKeyboard shortcutsGitHub Command PaletteWriting on GitHubStart writing on GitHubQuickstartAbout writing & formattingBasic formatting syntaxWork with advanced formattingOrganized data with tablesCollapsed sectionsCreate code blocksCreate diagramsMathematical expressionsAuto linked referencesAttaching filesAbout task listsPermanent links to codeUsing keywords in issues and pull requestsWork with saved repliesAbout saved repliesCreating a saved replyEditing a saved replyDeleting a saved replyUsing saved repliesShare content with gistsCreating gistsForking and cloning gistsSaving gists with starsExplore projectsContribute to open sourceContribute to a projectSave repositories with starsFollowing peopleFollowing organizationsGetting started with GitSet your usernameCaching credentialsGit passwordsmacOS Keychain credentialsGit workflowsAbout remote repositoriesManage remote repositoriesAssociate text editorsHandle line endingsIgnoring filesUsing GitAbout GitPush commits to a remoteGet changes from a remoteNon-fast-forward errorSplitting a subfolderAbout Git subtree mergesAbout Git rebaseGit rebaseResolve conflicts after rebaseSpecial characters in namesMaximum push limitSubversionSubversion & Git differencesSupport for Subversion clientsProperties supported by GitHubExploring integrationsAbout using integrationsAbout building integrationsFeatured integrationsGitHub Developer ProgramArchive account and public reposRequest account archiveGitHub Archive programUsing GitHub DocsDocs versionsHover cards"
    string_to_remove2 = "Help and supportDid this doc help you?YesNoPrivacy policyHelp us make these docs great!All GitHub docs are open source. See something that's wrong or unclear? Submit a pull request.Make a contributionLearn how to contributeStill need help?Ask the GitHub communityContact supportLegal© 2023 GitHub, Inc.TermsPrivacyStatusPricingExpert servicesBlog"

    interim_string = my_str.replace(string_to_remove1, "").replace(
        string_to_remove2, ""
    )

    # remove all useless whitespace in the string
    clean_string = (re.sub(" +", " ", (interim_string.replace("\n", " ")))).strip()
    # Pattern to match a period followed by a capital letter
    pattern = r"\.([A-Z])"
    # Replacement pattern - a period, a space, and the matched capital letter
    replacement = r". \1"

    # Substitute the pattern in the text with the replacement pattern
    return re.sub(pattern, replacement, clean_string)


def load_docs(url, max_depth=3):
    loader = RecursiveUrlLoader(
        url=url,
        max_depth=max_depth,
        extractor=lambda x: Soup(x, "html.parser").text,
        exclude_dirs=[
            "https://docs.github.com/en/enterprise-cloud@latest",
            "https://docs.github.com/en/enterprise-server@3.11",
            "https://docs.github.com/en/enterprise-server@3.10",
            "https://docs.github.com/en/enterprise-server@3.9",
            "https://docs.github.com/en/enterprise-server@3.8",
            "https://docs.github.com/en/enterprise-server@3.7",
        ],
    )
    return loader.load()


def flatten_list_of_lists(list_of_lists):
    # This function takes a list of lists (nested list) as input.
    # It returns a single flattened list containing all the elements
    # from the sublists, maintaining their order.

    # Using a list comprehension, iterate through each sublist in the list of lists.
    # For each sublist, iterate through each element.
    # The element is then added to the resulting list.
    return [element for sublist in list_of_lists for element in sublist]


def clean_docs(url_docs):
    cleaned_docs = [
        remove_extra_whitespace(element.page_content.replace("\n", ""))
        for element in url_docs
    ]
    metadata = [document.metadata for document in url_docs]

    return [
        Document(page_content=cleaned_docs[i], metadata=metadata[i])
        for i in range(len(url_docs))
    ]


def split_docs(documents, chunk_size=400, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    return text_splitter.split_documents(documents)
