"""
run_ui.py - Standalone Evaluator UI launcher for Laptop 2.
Launches a lightweight local server and opens the PathSnap Evaluator Dashboard in the default web browser.
"""

import http.server
import socketserver
import webbrowser
import os

PORT = 8501

class EvaluatorHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            self.path = "/templates/index.html"
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"============================================================")
    print(f"PATHSNAP EVALUATOR INTERFACE (LAPTOP 2)")
    print(f"Serving dashboard at: http://localhost:{PORT}")
    print(f"Configure Laptop 1's API URL in the UI header input box.")
    print(f"============================================================")
    
    # Try to open in browser
    try:
        webbrowser.open(f"http://localhost:{PORT}")
    except Exception:
        pass
        
    with socketserver.TCPServer(("", PORT), EvaluatorHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down Evaluator UI...")

