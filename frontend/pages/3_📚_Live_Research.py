import streamlit as st
import requests
import urllib.parse
import re

st.set_page_config(page_title="Live Research", layout="wide")
st.title("📚 Live Academic Research Explorer")
st.write("Search the Crossref database for the latest peer-reviewed papers from global academic journals.")

search_query = st.text_input("Enter a concept or problem to research:", value="Quantum Support Vector Machine")
max_results = st.slider("Number of papers to fetch", 1, 10, 5)

if st.button("Search Academic Journals"):
    with st.spinner(f"Searching global databases for '{search_query}'..."):
        
        formatted_query = urllib.parse.quote(search_query)
        
        # Using Crossref API - highly reliable for academic metadata
        url = f"https://api.crossref.org/works?query={formatted_query}&select=title,abstract,URL,published,author&rows={max_results}&sort=relevance"

        try:
            # THE FIX: The 'mailto' puts you in Crossref's "Polite Pool", preventing 429 rate limit errors!
            headers = {'User-Agent': 'QSVM_Student_Project/1.0 (mailto:studentproject@example.com)'}
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get("message", {}).get("items", [])
                
                if not items:
                    st.warning(f"No papers found for '{search_query}'. Try broader terms!")
                else:
                    st.success("Research papers successfully retrieved!")
                    
                    for paper in items:
                        # Safely extract the title
                        title_list = paper.get("title", [])
                        title = title_list[0] if title_list else "Unknown Title"
                        
                        url_link = paper.get("URL", "#")
                        abstract = paper.get("abstract", "No abstract provided by the publisher.")
                        
                        # Clean up abstract if it has XML tags (Crossref sometimes includes <jats:p>)
                        abstract = re.sub(r'<[^>]+>', '', abstract)
                        
                        # Extract authors
                        authors = paper.get("author", [])
                        author_names = ", ".join([f"{a.get('given', '')} {a.get('family', '')}".strip() for a in authors]) if authors else "Unknown Authors"
                        
                        with st.expander(f"📄 {title}"):
                            st.markdown(f"**Authors:** *{author_names}*")
                            st.write("**Abstract:**")
                            st.write(abstract if abstract else "No abstract available.")
                            st.markdown(f"[🔗 View Publication on Publisher Site]({url_link})")
                            
            elif response.status_code == 429:
                st.error("Rate limit reached. Please wait 60 seconds and try again.")
            else:
                st.error(f"API Error: Server returned status code {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Connection Error: {e}")
            st.info("Check your internet connection or try again later.")