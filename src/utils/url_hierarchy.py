from urllib.parse import urlparse, unquote


def _host_label(netloc: str) -> str:
    host = netloc.lower()
    parts = host.split('.')
    # cds.iisc.ac.in -> "iisc cds"
    if host.endswith('iisc.ac.in'):
        labels = []
        if 'iisc' in parts:
            labels.append('iisc')
        if 'cds' in parts:
            labels.append('cds')
        if labels:
            return ' '.join(labels)
    # *.github.io -> subdomain as label (e.g., at-cg.github.io -> "at cg")
    if host.endswith('github.io') and len(parts) >= 3:
        sub = parts[0].replace('-', ' ').strip()
        return sub or 'github io'
    # Default: strip TLD and join remaining
    if len(parts) > 2:
        return ' '.join(p for p in parts[:-2] if p)
    if len(parts) > 1:
        return parts[0]
    return host


def parse_url_hierarchy(url: str):
    """Return a dict with host_label and path_segments (normalized).

    - host_label: user-friendly root grouping label derived from hostname
    - path_segments: cleaned URL path parts (hyphens to spaces, percent-decoded)
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return {"host_label": "", "path_segments": []}
    host = parsed.netloc
    host_label = _host_label(host)
    raw_segs = [s for s in parsed.path.split('/') if s]
    segs = [unquote(s).replace('-', ' ').strip() for s in raw_segs]
    return {"host_label": host_label, "path_segments": segs}


def build_url_heading_path(url: str):
    """Build a heading-like path from URL components as list of {level, text}.

    Example: https://cds.iisc.ac.in/people/chirag-jain/
      -> [{level:1, text:"iisc cds"}, {level:2, text:"people"}, {level:3, text:"chirag jain"}]
    """
    info = parse_url_hierarchy(url)
    path = []
    level = 1
    if info["host_label"]:
        path.append({"level": level, "text": info["host_label"]})
        level += 1
    for seg in info["path_segments"]:
        if not seg:
            continue
        path.append({"level": level, "text": seg})
        level += 1
    return path
