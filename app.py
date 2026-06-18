import os
# Force matplotlib to use the writable /tmp directory for its cache
# Critical for serverless environments like Vercel
os.environ["MPLCONFIGDIR"] = "/tmp"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import re
import io
import time
import json
import zipfile
import textwrap
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from google import genai
import pandas as pd
import jinja2

load_dotenv()

app = Flask(__name__, template_folder='templates')

GEMINI_MODEL = "gemini-2.5-flash-lite"
SECTION_TAGS = [
    "ABSTRACT", "INTRODUCTION", "RELATED_WORK", "METHODOLOGY",
    "RESULTS_AND_ANALYSIS", "DISCUSSION", "CONCLUSION", "ACKNOWLEDGEMENTS",
]
WORDS_PER_PAGE_SINGLE = 450
WORDS_PER_PAGE_DOUBLE = 600
SECTION_WEIGHT = {
    "ABSTRACT": 0.05, "INTRODUCTION": 0.16, "RELATED_WORK": 0.14,
    "METHODOLOGY": 0.22, "RESULTS_AND_ANALYSIS": 0.20, "DISCUSSION": 0.12,
    "CONCLUSION": 0.07, "ACKNOWLEDGEMENTS": 0.04,
}

# --- Utilities ---

def escape_for_latex(text: str) -> str:
    text = str(text)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def compute_section_targets(mode: str, value: int, two_col: bool) -> dict:
    if mode == "pages":
        wpp = WORDS_PER_PAGE_DOUBLE if two_col else WORDS_PER_PAGE_SINGLE
        total_words = value * wpp
    else:
        total_words = value
        
    return {tag: max(80, int(total_words * w)) for tag, w in SECTION_WEIGHT.items()}

def fetch_arxiv_literature(query: str, max_results: int = 6):
    if not query.strip(): query = "machine learning"
    safe_query = urllib.parse.quote(query)
    url = f"https://export.arxiv.org/api/query?search_query=all:{safe_query}&start=0&max_results={max_results}&sortBy=relevance"
    
    bibtex_entries, abstract_summaries = [], []
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            xml_data = response.read()
        root = ET.fromstring(xml_data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text.replace('\n', ' ').strip()
            summary = entry.find('atom:summary', ns).text.replace('\n', ' ').strip()
            published = entry.find('atom:published', ns).text[:4]
            authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]
            author_str = " and ".join(authors)
            
            last_name = authors[0].split()[-1].lower() if authors else "unknown"
            first_word = re.sub(r'[^a-zA-Z0-9]', '', title.split()[0].lower())
            cite_key = f"{last_name}{published}{first_word}"
            
            bibtex = textwrap.dedent(f"""
            @article{{{cite_key},
              title={{{title}}},
              author={{{author_str}}},
              journal={{arXiv preprint}},
              year={{{published}}}
            }}""").strip()
            bibtex_entries.append(bibtex)
            abstract_summaries.append(f"[{cite_key}] {title} ({published}): {summary}")
        return "\n\n".join(bibtex_entries), "\n\n".join(abstract_summaries)
    except Exception:
        return "", ""

def call_gemini_with_retry(api_key: str, prompt: str, retries: int = 3) -> str:
    client = genai.Client(api_key=api_key)
    for attempt in range(retries):
        try:
            res = client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt,
                config=genai.types.GenerateContentConfig(temperature=0.2)
            )
            return res.text
        except Exception as e:
            if "503" in str(e) or "429" in str(e):
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
            raise e
    return ""

def clean_llm_latex_output(raw_output: str) -> str:
    clean_text = raw_output.strip()
    clean_text = re.sub(r'^```[a-zA-Z]*\n', '', clean_text)
    clean_text = re.sub(r'\n```$', '', clean_text)
    return clean_text.strip()

def generate_section_prompt(tag, title, arxiv_context, bibtex, user_notes, venue, previous_sections, target_words):
    history = ""
    if previous_sections:
        history = "PREVIOUSLY GENERATED SECTIONS:\n"
        for k, v in previous_sections.items():
            history += f"--- {k} ---\n{v[:500]}... [truncated]\n\n"
            
    cite_keys = re.findall(r"@\w+\{(\w+),", bibtex)
    cite_str = ", ".join(cite_keys) if cite_keys else "(none provided)"

    return textwrap.dedent(f"""
    You are an elite academic researcher writing a paper for {venue}. 
    You are writing ONE specific section of a research paper.
    PAPER TITLE: {title}
    CURRENT SECTION TO WRITE: {tag}
    TARGET WORD COUNT: ~{target_words} words.
    REAL LITERATURE CONTEXT: {arxiv_context}
    AVAILABLE CITATION KEYS: {cite_str}
    USER NOTES FOR THIS PAPER: {user_notes}
    {history}
    INSTRUCTIONS:
    1. Write ONLY the '{tag}' section. Do not write any other sections.
    2. Write in formal, third-person academic LaTeX prose suitable for {venue}.
    3. Use \\cite{{key}} frequently and accurately based on the Literature Context.
    4. Do not output markdown code fences. Output raw text.
    5. Do not output the section header (e.g., no \\section{{{tag}}}).
    6. Ensure the narrative flows logically from the previously generated sections.
    """).strip()

def generate_academic_chart(api_key: str, paper_title: str, results_notes: str, figure_index: int):
    client = genai.Client(api_key=api_key)
    
    # CRITICAL VERCEL FIX: Must write to /tmp
    chart_filename = f"/tmp/generated_chart_{figure_index}.png" 
    
    prompt = f"""
    Write an isolated Python script using matplotlib to generate a publication-quality chart.
    Title: {paper_title}
    Data: {results_notes}
    Target Figure Filename: {chart_filename}
    REQUIREMENTS:
    1. Output ONLY raw executable Python code. No markdown.
    2. Save file exactly as '{chart_filename}' using plt.savefig.
    3. Use plt.close('all') at the end. Do not use plt.show().
    """
    try:
        res = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt,
            config=genai.types.GenerateContentConfig(temperature=0.1)
        )
        clean_code = res.text.strip().replace("```python", "").replace("```", "")
        local_scope = {"plt": plt, "pd": pd, "re": re}
        plt.close('all')
        
        # Execute the LLM's code
        exec(clean_code, globals(), local_scope)
        
        # Read the file back from /tmp
        if os.path.exists(chart_filename):
            with open(chart_filename, "rb") as f:
                img_bytes = f.read()
            os.remove(chart_filename) # Clean up the tmp file
            plt.close('all')
            # Return just the base filename for the zip/latex process
            return f"generated_chart_{figure_index}.png", img_bytes
    except Exception as e:
        print(f"Chart Gen Error: {e}")
        plt.close('all')
        return None

LATEX_TMPL = r"""
\documentclass[11pt[% if two_col %],twocolumn[% endif %]]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{geometry}
\geometry{[% if two_col %]letterpaper,margin=0.7in,columnsep=18pt[% else %]letterpaper,margin=1in[% endif %]}
\usepackage{graphicx,hyperref,titlesec,authblk,xcolor}
\usepackage[numbers]{natbib}
\title{\vspace{-1.5em}\Large\bfseries [[ escaped_title ]]%
[% if kw_raw %]\\\vspace{0.3em}\normalsize\textit{Keywords:}\ \small [[ escaped_kw ]][% endif %]}
[% for a in authors %]
\author[[[ loop.index ]]]{\textbf{[[ a.name ]]}[% if a.email %]\thanks{\href{mailto:[[ a.email ]]}{[[ a.email ]]}}[% endif %]}
\affil[[[ loop.index ]]]{\small\textit{[[ a.affil ]]}}
[% endfor %]
\date{\today}
\begin{document}
\maketitle
\begin{abstract}\noindent [[ sec.ABSTRACT ]]\end{abstract}
\section{Introduction}\label{sec:intro}
[[ sec.INTRODUCTION ]]
\section{Related Work}\label{sec:related}
[[ sec.RELATED_WORK ]]
\section{Methodology}\label{sec:method}
[[ sec.METHODOLOGY ]]
[% if figs %]
[% for f in figs %]
\begin{figure}[ht]\centering
\includegraphics[width=0.95\linewidth]{[[ f.fn ]]}
\caption{[[ f.cap ]]}\label{fig:[[ loop.index ]]}
\end{figure}
[% endfor %]
[% endif %]
\section{Results and Analysis}\label{sec:results}
[[ sec.RESULTS_AND_ANALYSIS ]]
\section{Discussion}\label{sec:disc}
[[ sec.DISCUSSION ]]
\section{Conclusion}\label{sec:concl}
[[ sec.CONCLUSION ]]
\section*{Acknowledgements}
[[ sec.ACKNOWLEDGEMENTS ]]
[% if bibtex %]
\begin{filecontents*}{\jobname.bib}
[[ bibtex ]]
\end{filecontents*}
\bibliographystyle{unsrtnat}
\bibliography{\jobname}
[% endif %]
\end{document}
"""

def render_latex(title, authors_list, two_col, keywords, sections, bibtex, figs=None):
    env = jinja2.Environment(
        block_start_string="[%", block_end_string="%]",
        variable_start_string="[[", variable_end_string="]]",
        autoescape=False
    )
    tmpl = env.from_string(LATEX_TMPL)
    
    clean_authors = []
    for a in authors_list:
        clean_authors.append({
            "name": escape_for_latex(a.get("name","")),
            "affil": escape_for_latex(a.get("affil","")),
            "email": escape_for_latex(a.get("email","")),
        })

    return tmpl.render(
        escaped_title=escape_for_latex(title),
        escaped_kw=escape_for_latex(keywords),
        kw_raw=keywords.strip(),
        authors=clean_authors,
        two_col=two_col,
        sec=sections,
        bibtex=bibtex.strip(),
        figs=figs or []
    )

def build_zip(latex_src, bibtex_data, generated_figs, uploaded_files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("main.tex", latex_src)
        if bibtex_data.strip():
            zf.writestr("references.bib", bibtex_data.strip())
        
        # Write LLM generated charts
        for fname, fbytes in generated_figs:
            zf.writestr(fname, fbytes)
            
        # Write User Uploaded Figures
        for file in uploaded_files:
            if file.filename:
                safe_name = secure_filename(file.filename)
                file.seek(0)
                zf.writestr(safe_name, file.read())
                
    buf.seek(0)
    return buf

# --- Routes ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/generate", methods=["POST"])
def generate():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return jsonify({"error": "Gemini API Key missing in environment configuration."}), 400

    title = request.form.get("title", "Untitled Paper")
    keywords = request.form.get("keywords", "")
    venue = request.form.get("venue", "Custom / Unspecified")
    layout = request.form.get("layout", "single")
    two_col = (layout == "double")
    
    length_mode = request.form.get("length_mode", "pages")
    length_val = int(request.form.get("length_val", 4 if length_mode == 'pages' else 2500))
    
    user_bibtex = request.form.get("bibtex", "")
    
    # Parse JSON strings passed via form-data
    try:
        authors = json.loads(request.form.get("authors", "[]"))
        notes = json.loads(request.form.get("notes", "{}"))
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON mapping for authors or notes"}), 400
    
    uploaded_files = request.files.getlist('figures')

    sec_targets = compute_section_targets(length_mode, length_val, two_col)
    compiled_notes = f"Abstract: {notes.get('abstract','')}\nIntro: {notes.get('intro','')}\nMethod: {notes.get('method','')}\nResults: {notes.get('results','')}\nExtra: {notes.get('extra','')}"

    try:
        # 1. Fetch Literature
        real_bibtex, arxiv_summaries = fetch_arxiv_literature(f"{title} {keywords}", 6)
        combined_bibtex = f"{user_bibtex}\n\n{real_bibtex}".strip()

        # 2. Generate Chart
        generated_charts = []
        if notes.get("results"):
            chart = generate_academic_chart(api_key, title, notes.get("results"), 1)
            if chart: generated_charts.append(chart)

        # 3. Generate Sections (Sequential)
        sections_dict = {}
        for tag in SECTION_TAGS:
            prompt = generate_section_prompt(
                tag, title, arxiv_summaries, combined_bibtex, compiled_notes, venue,
                sections_dict, sec_targets.get(tag, 300)
            )
            out = call_gemini_with_retry(api_key, prompt)
            sections_dict[tag] = clean_llm_latex_output(out)

        # 4. Compile Figure List for LaTeX
        fig_list = []
        for file in uploaded_files:
            if file.filename:
                fig_list.append({"fn": secure_filename(file.filename), "cap": f"Uploaded observation: {file.filename}"})
        for fname, _ in generated_charts:
            fig_list.append({"fn": fname, "cap": "Programmatic analytical verification."})

        # 5. Render and ZIP
        latex_src = render_latex(title, authors, two_col, keywords, sections_dict, combined_bibtex, fig_list)
        zip_buf = build_zip(latex_src, combined_bibtex, generated_charts, uploaded_files)
        
        return send_file(
            zip_buf,
            mimetype="application/zip",
            as_attachment=True,
            download_name="sciwrite_project.zip"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)