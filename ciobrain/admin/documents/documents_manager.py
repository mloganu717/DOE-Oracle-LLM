"""
ciobrain/admin/documents/documents_manager.py

Classes:
    - DocumentsManager: manages documents operations
    - StorageHandler: used by DocumentsManager to handle file operations
"""

import os
from flask import current_app
from werkzeug.utils import secure_filename

class DocumentsManager:
    """Manages document operations"""

    def __init__(self):
        self.storage_handler = StorageHandler()
        self.research_papers_dir = None

    def initialize(self, app):
        """Initialize paths using the Flask app context"""
        self.research_papers_dir = os.path.join(app.config['KNOWLEDGE'], 'research_papers')
        self.storage_handler.initialize(app)

    def get_directories(self) -> dict[str, list[str]]:
        """Get directory contents using the StorageHandler"""
        return self.storage_handler.get_directory_contents()

    def upload_document(self, file, is_research_paper=False) -> str:
        """Handle document upload using StorageHandler"""
        return self.storage_handler.process_upload(file, is_research_paper)

class StorageHandler:
    """Handles validation and file operations"""

    def __init__(self):
        self.research_papers_dir = None

    def initialize(self, app):
        """Initialize paths using the Flask app context"""
        self.research_papers_dir = os.path.join(app.config['KNOWLEDGE'], 'research_papers')
        os.makedirs(self.research_papers_dir, exist_ok=True)

    def get_directory_contents(self) -> dict[str, list[str]]:
        """Retrieve the contents of the storage directories."""
        if not self.research_papers_dir:
            return {'research_papers': []}
            
        try:
            return {
                'research_papers': os.listdir(self.research_papers_dir)
            }
        except FileNotFoundError:
            return {'research_papers': []}

    def process_upload(self, file, is_research_paper=False) -> str:
        """validate and save a file in one call."""
        self._validate_file(file)
        return self._save_file(file, is_research_paper)

    def _validate_file(self, file) -> None:
        """Validate the file extension based on allowed extensions."""
        allowed_extensions = self._get_allowed_extensions()
        file_extension = os.path.splitext(file.filename)[1].lower()
        if '.' not in file.filename or file_extension not in allowed_extensions:
            raise ValueError("Unsuported file type")

    def _save_file(self, file, is_research_paper=False) -> str:
        """Save validated file to appropriate directory."""
        if not self.research_papers_dir:
            raise ValueError("StorageHandler not properly initialized")
            
        filename = secure_filename(file.filename)
        file_path = os.path.join(self.research_papers_dir, filename)
        file.save(file_path)
        return file_path

    def _get_allowed_extensions(self) -> set[str]:
        """Retrieve allowed extensions from config."""
        return {'.pdf'}  # Only allow PDF files for research papers
