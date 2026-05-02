import urllib.request
import json

boundary = '----TestBoundary123'
CRLF = '\r\n'

def encode_multipart(fields, boundary):
    parts = []
    for name, value in fields.items():
        part = (
            f'--{boundary}' + CRLF +
            f'Content-Disposition: form-data; name="{name}"' + CRLF +
            CRLF +
            value + CRLF
        )
        parts.append(part)
    parts.append(f'--{boundary}--' + CRLF)
    return ''.join(parts).encode('utf-8')

fields = {
    'jd_text': 'We are looking for a Python developer with experience in FastAPI, Docker, and SQL. Knowledge of React is a plus.',
    'resume_text': 'Experienced Python developer with 3 years of experience in Django, Docker, and React. Familiar with REST APIs and PostgreSQL.'
}

body = encode_multipart(fields, boundary)

req = urllib.request.Request(
    'http://localhost:8000/analyze',
    data=body,
    headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
    method='POST'
)

try:
    r = urllib.request.urlopen(req, timeout=180)
    data = json.loads(r.read())
    print('Status:', r.status)
    print('Similarity:', data.get('similarity_score'))
    print('Matched:', data.get('matched_skills'))
    print('Missing:', data.get('missing_skills'))
    print('Feedback:', data.get('feedback'))
    print('Section scores:', data.get('section_scores'))
    print('\n=== /analyze PASSED ===')
except urllib.error.HTTPError as e:
    print('HTTPError:', e.code)
    print(e.read().decode())
except Exception as ex:
    print('Error:', type(ex).__name__, ex)
