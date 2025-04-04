from flask import Blueprint, render_template, request, redirect, url_for, flash
from ciobrain.admin.documents.documents_manager import DocumentsManager
from ciobrain.admin.documents.rag_manager import RAGManager

documents_bp = Blueprint('documents', __name__)
documents_manager = DocumentsManager()
rag_manager = RAGManager()

@documents_bp.route('/documents')
def home():
    """Document management page"""
    directories = documents_manager.get_directories()
    return render_template('admin_documents.html', directories=directories)

@documents_bp.route('documents/upload', methods=['POST'])
def upload_document():
    """Document upload operation"""
    file = request.files.get('file')
    is_research_paper = request.form.get('is_research_paper', 'false').lower() == 'true'
    
    if file:
        try:
            documents_manager.upload_document(file, is_research_paper=is_research_paper)
            flash("File uploaded successfully!", "success")
        except ValueError as e:
            flash(str(e), "error")
    else:
        flash("No file selected!", "error")
    return redirect(url_for('admin.documents.home'))

@documents_bp.route('documents/process', methods=['POST'])
def process_documents():
    """Process all research papers"""
    vector_db = rag_manager.process_documents()
    if vector_db:
        flash("Documents processed successfully!", "success")
        return redirect(url_for('admin.documents.home'))
    else:
        flash("Error processing documents", "error")
        return redirect(url_for('admin.documents.home'))

