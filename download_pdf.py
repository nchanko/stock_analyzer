import streamlit as st
from fpdf import FPDF

# Define a function to create a PDF
def create_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=text, ln=True, align="C")
    # Save the PDF to a temporary file and return its file path
    pdf_file_path = "result.pdf"
    pdf.output(pdf_file_path)
    return pdf_file_path

# Example usage in Streamlit
text_to_include = "This is an example text to include in the PDF."

# When the button is clicked, generate the PDF and offer it for download
if st.button('Download Results as PDF'):
    pdf_file_path = create_pdf(text_to_include)
    with open(pdf_file_path, "rb") as pdf_file:
        st.download_button(label="Download PDF",
                           data=pdf_file,
                           file_name="results.pdf",
                           mime="application/pdf")
