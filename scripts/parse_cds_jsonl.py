import json
import re
import csv
import pandas as pd

def extract_faculty(text):
    faculty = []
    # Look for faculty section with multiple patterns
    patterns = [
        r'Faculty\n(.*?)(Explore|Labs:|Students|Alumni|Contact Us)',
        r'Faculty Members\n(.*?)(Research|Labs:|Students)',
        r'People\n.*?Faculty\n(.*?)(Students|Research)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            lines = match.group(1).split('\n')
            current_faculty = {}
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Email detection
                if '@' in line and '.' in line:
                    current_faculty['email'] = line
                    if current_faculty.get('name'):
                        faculty.append(current_faculty.copy())
                        current_faculty = {}
                # Title detection (Prof, Dr, Assistant Professor, etc.)
                elif re.search(r'(Prof|Dr|Assistant|Associate|Professor)', line, re.IGNORECASE):
                    current_faculty['title'] = line
                # Name detection (capitalize words, no special chars)
                elif re.match(r'^[A-Z][a-zA-Z\s\.]+$', line) and len(line.split()) <= 4:
                    current_faculty['name'] = line
            break
    return faculty

def extract_students(text):
    students = []
    patterns = [
        r'Current Students\nPh\.D Students\n(.*?)(M\.Tech|Recent News|Contact Us)',
        r'PhD Students\n(.*?)(Masters|M\.Tech|Contact)',
        r'Students\n(.*?)(Faculty|Research|Contact)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            lines = match.group(1).split('\n')
            for line in lines:
                # Tab-separated or multiple spaces
                parts = re.split(r'\t+|\s{3,}', line.strip())
                if len(parts) >= 3:
                    students.append({
                        "name": parts[0].strip(),
                        "program": parts[1].strip() if len(parts) > 1 else '',
                        "stream": parts[2].strip() if len(parts) > 2 else '',
                        "advisor": parts[3].strip() if len(parts) > 3 else ''
                    })
            break
    return students

def extract_research_areas(text):
    areas = []
    patterns = [
        r'Research\n(.*?)(Labs:|Recent News|Contact Us)',
        r'Research Areas\n(.*?)(Labs:|Faculty|Contact)',
        r'Research Interests\n(.*?)(Publications|Faculty|Contact)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            lines = match.group(1).split('\n')
            for line in lines:
                line = line.strip()
                if line and len(line) > 3 and not line.startswith('Recent'):
                    areas.append(line)
            break
    return areas

def extract_labs(text):
    labs = []
    patterns = [
        r'Labs:\n(.*?)(Recent News|Contact Us|\n\n)',
        r'Research Groups\n(.*?)(Contact|Faculty)',
        r'Laboratories\n(.*?)(Contact|News)'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            lines = match[0].split('\n')
            for line in lines:
                # Match lab name with faculty in parentheses
                lab_match = re.match(r'(.+?)\s*\((Prof\. .+?|Dr\. .+?)\)', line.strip())
                if lab_match:
                    labs.append({
                        "lab": lab_match.group(1).strip(),
                        "faculty": lab_match.group(2).replace('Prof. ', '').replace('Dr. ', '').strip()
                    })
    return labs

def extract_publications(text):
    pubs = []
    # More sophisticated publication extraction
    pub_keywords = ['proceedings', 'journal', 'conference', 'paper', 'publication', 
                   'ieee', 'acm', 'springer', 'award', 'citation']
    
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        # Check for publication-like patterns
        if any(keyword in line.lower() for keyword in pub_keywords):
            # Additional filtering for meaningful content
            if len(line) > 20 and any(char.isdigit() for char in line):
                pubs.append(line)
        # DOI pattern
        elif re.search(r'10\.\d+/', line):
            pubs.append(line)
        # Year pattern with authors
        elif re.search(r'\b(19|20)\d{2}\b.*[A-Z][a-z]+.*[A-Z][a-z]+', line):
            pubs.append(line)
    
    return list(set(pubs))  # Remove duplicates

def extract_contact(text):
    contact = {}
    patterns = [
        r'Contact Us\n(.*?)(Follow us|Locate us|Submit Content)',
        r'Contact\n(.*?)(Address|Phone|Email)',
        r'Contact Information\n(.*?)(\n\n|$)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            contact_text = match.group(1)
            
            # Extract phone numbers
            phone = re.search(r'Phone:\s*([+\d\s\-\(\)]+)', contact_text)
            if phone:
                contact['phone'] = phone.group(1).strip()
            
            # Extract emails
            emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', contact_text)
            if emails:
                contact['email'] = ', '.join(emails)
            
            # Extract address
            addr = re.search(r'Address:\s*([^\n]+)', contact_text)
            if addr:
                contact['address'] = addr.group(1).strip()
            break
    
    return contact

def flatten_data_for_csv(structured_data):
    """Flatten nested JSON structure for CSV output"""
    flattened = []
    
    for entry in structured_data:
        base_row = {
            'url': entry.get('url', ''),
            'contact_phone': entry.get('contact', {}).get('phone', ''),
            'contact_email': entry.get('contact', {}).get('email', ''),
            'contact_address': entry.get('contact', {}).get('address', ''),
            'research_areas': '; '.join(entry.get('research_areas', [])),
            'publications_count': len(entry.get('publications', [])),
            'publications': ' | '.join(entry.get('publications', []))[:500]  # Truncate for CSV
        }
        
        # Faculty rows
        for faculty in entry.get('faculty', []):
            row = base_row.copy()
            row.update({
                'type': 'faculty',
                'name': faculty.get('name', ''),
                'title': faculty.get('title', ''),
                'email': faculty.get('email', ''),
                'program': '',
                'stream': '',
                'advisor': '',
                'lab': ''
            })
            flattened.append(row)
        
        # Student rows
        for student in entry.get('students', []):
            row = base_row.copy()
            row.update({
                'type': 'student',
                'name': student.get('name', ''),
                'title': '',
                'email': '',
                'program': student.get('program', ''),
                'stream': student.get('stream', ''),
                'advisor': student.get('advisor', ''),
                'lab': ''
            })
            flattened.append(row)
        
        # Lab rows
        for lab in entry.get('labs', []):
            row = base_row.copy()
            row.update({
                'type': 'lab',
                'name': lab.get('lab', ''),
                'title': '',
                'email': '',
                'program': '',
                'stream': '',
                'advisor': lab.get('faculty', ''),
                'lab': lab.get('lab', '')
            })
            flattened.append(row)
    
    return flattened

def main():
    input_file = "cds.jsonl"
    json_output_file = "cds_structured.json"
    csv_output_file = "cds_structured.csv"
    
    structured_data = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                text = entry.get("text", "")
                
                structured_entry = {
                    "url": entry.get("url"),
                    "title": entry.get("title", ""),
                    "faculty": extract_faculty(text),
                    "students": extract_students(text),
                    "research_areas": extract_research_areas(text),
                    "labs": extract_labs(text),
                    "publications": extract_publications(text),
                    "contact": extract_contact(text)
                }
                structured_data.append(structured_entry)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line")
                continue

    # Save as JSON
    with open(json_output_file, "w", encoding="utf-8") as out:
        json.dump(structured_data, out, indent=2, ensure_ascii=False)

    # Save as CSV
    flattened_data = flatten_data_for_csv(structured_data)
    df = pd.DataFrame(flattened_data)
    df.to_csv(csv_output_file, index=False, encoding="utf-8")
    
    print(f"Structured data saved to {json_output_file}")
    print(f"CSV data saved to {csv_output_file}")
    print(f"Total entries processed: {len(structured_data)}")
    print(f"Total rows in CSV: {len(flattened_data)}")

if __name__ == "__main__":
    main()