import http.server
import socketserver
import json

PORT = 8001

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/data':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            try:
                with open("gauge_value.json", "r") as f:
                    data = json.load(f)
            except FileNotFoundError:
                data = {"value": 0}  # default value if file not found
            self.wfile.write(json.dumps(data).encode())
        else:
            super().do_GET()

def run_server():
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving at port {PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()