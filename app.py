import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled
from langchain.document_loaders import UnstructuredURLLoader
from langchain.schema import Document
from bs4 import BeautifulSoup
import requests
import os
from dotenv import load_dotenv

# Streamlit app setup
st.set_page_config(page_title="LangChain: Summarize Text from YouTube or Website", page_icon="🐦")
st.title("🐦 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

# Load environment variables
load_dotenv()

# Sidebar API key input
with st.sidebar:
    groq_api_key = st.text_input("Groq_api_key", value="", type="password")

# URL input
url = st.text_input("URL ", label_visibility="collapsed")

# LLM setup
llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)

# Prompt template
prompt_template = """
Provide a summary of the following content in 300 words:
content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['text'])

if st.button("Summarize the content from YT or Website"):
    if not groq_api_key.strip() or not url.strip():
        st.error("⚠️ Please provide both API key and URL.")
    elif not validators.url(url):
        st.error("⚠️ Please enter a valid URL (YouTube or website).")
    else:
        try:
            with st.spinner("Fetching transcript/content and summarizing..."):
                docs = []

                # Check if YouTube URL
                if "youtube.com" in url or "youtu.be" in url:
                    st.video(url)
                    if "youtu.be" in url:
                        video_id = url.split("/")[-1]
                    else:
                        video_id = url.split("v=")[-1].split("&")[0]

                    # Try to get transcript
                    try:
                        transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
                        transcript_text = " ".join([t["text"] for t in transcript_data])
                        docs = [Document(page_content=transcript_text)]
                    except (NoTranscriptFound, TranscriptsDisabled):
                        st.warning("⚠️ Transcript not available. Using video description as fallback.")
                        # Fallback: fetch video page and scrape description
                        page = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
                        if page.status_code == 200:
                            soup = BeautifulSoup(page.text, "html.parser")
                            description_tag = soup.find("meta", {"name": "description"})
                            if description_tag and description_tag.get("content"):
                                description = description_tag.get("content")
                                docs = [Document(page_content=description)]
                            else:
                                st.error("❌ Could not extract video description.")
                        else:
                            st.error("❌ Failed to fetch YouTube page.")

                else:
                    # Handle normal websites
                    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
                    if response.status_code == 200 and response.text.strip():
                        loader = UnstructuredURLLoader(
                            urls=[url],
                            ssl_verify=False,
                            headers={
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                                "Accept-Language": "en-US"
                            }
                        )
                        try:
                            docs = loader.load()
                        except Exception as e:
                            st.error(f"❌ Failed to parse website content: {e}")
                    else:
                        st.error("❌ Website returned empty content or is inaccessible.")

                # Run summarization if docs exist
                if docs:
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    summary = chain.run(docs)
                    st.subheader("📝 Summary")
                    st.success(summary)
                else:
                    st.warning("⚠️ No content to summarize.")

        except Exception as e:
            st.error(f"❌ Exception: {e}")
