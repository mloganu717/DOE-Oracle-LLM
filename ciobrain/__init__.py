"""
ciobrain/__init__.py

This module initialized the Flask application, sets up configuration settings,
creates necessary directories, and registers the main Blueprints for the admin
and customer sections.
"""

import os
from flask import Flask, render_template, request, jsonify, redirect, url_for, Response, send_from_directory
from flask_cors import CORS, cross_origin
from ciobrain.mediator import Mediator
import subprocess
import logging
from mlx_lm import load, generate


def create_app(test_config=None):
    """Initialize and configure the Flask app instance"""

    app = Flask(__name__, instance_relative_config=True)
    CORS(app)  # Enable CORS for all routes

    # Ensure the 'instance' directory is created
    os.makedirs(app.instance_path, exist_ok=True)

    # Default configuration settings
    app.config.from_mapping(
        SECRET_KEY='dev', # Set your own
        DATABASE=os.path.join(app.instance_path, 'ciobrain.sqlite'),
        KNOWLEDGE=os.path.join(app.instance_path, 'knowledge'),
        VECTOR_STORE=os.path.join(app.instance_path, 'knowledge/vector_store')
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    directories = [
        app.config.get('KNOWLEDGE'),
        app.config.get('VECTOR_STORE')
    ]

    for directory in directories:
        if directory:
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as e:
                app.logger.error("Error creating directory %s: %s", directory, e)

    mediator = Mediator()

    # API endpoint for prompt handling
    @app.route('/prompt', methods=['POST', 'OPTIONS'])
    @cross_origin()
    def prompt():
        if request.method == 'OPTIONS':
            return Response('', 
                status=200,
                headers={
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type'
                }
            )
            
        try:
            data = request.get_json()
            if not data:
                return Response("No data provided", status=400)

            prompt = data.get('prompt', '').strip()
            use_rag = data.get('use_rag', True)

            if not prompt:
                logging.warning("Received empty prompt")
                return Response("Invalid prompt", status=400)

            logging.info(f"Received prompt: {prompt}, Use RAG: {use_rag}")

            # Convert the prompt into the format expected by the mediator
            messages = [{'role': 'user', 'content': prompt}]

            # Get the response generator from the mediator
            response_generator = mediator.stream(messages, use_rag=use_rag)

            # Return as plain text with the generator yielding properly encoded bytes
            return Response(
                response_generator,
                content_type="text/plain",
                headers={
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type'
                }
            )
        except Exception as e:
            logging.error(f"Error in /prompt route: {str(e)}")
            return Response(f"Internal server error: {str(e)}", status=500, 
                headers={
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type'
                }
            )

    # Serve frontend static files in production
    @app.route('/')
    def serve_frontend():
        return send_from_directory('../frontend/dist', 'index.html')

    @app.route('/<path:path>')
    def serve_static(path):
        return send_from_directory('../frontend/dist', path)
    
    with app.app_context():
        mediator.initialize_resources()

    from . import db
    db.init_app(app)

    return app