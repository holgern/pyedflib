#!/usr/bin/env python
"""
gh_lists.py MILESTONE

Functions for Github API requests.
"""
import argparse
import collections
import json
import os
import re
import sys

from urllib2 import urlopen

Issue = collections.namedtuple('Issue', ('id', 'title', 'url'))


def main():
    p = argparse.ArgumentParser(usage=__doc__.lstrip())
    p.add_argument('--project', default='holgern/pyedflib')
    p.add_argument('milestone')
    args = p.parse_args()

    getter = CachedGet('gh_cache.json')
    try:
        milestones = get_milestones(getter, args.project)
        if args.milestone not in milestones:
            msg = "Milestone {0} not available. Available milestones: {1}"
            msg = msg.format(args.milestone, ", ".join(sorted(milestones)))
            p.error(msg)
        issues = get_issues(getter, args.project, args.milestone)
        issues.sort()
    finally:
        getter.save()

    prs = [x for x in issues if '/pull/' in x.url]
    issues = [x for x in issues if x not in prs]

    def print_list(title, items):
        print()
        print(title)
        print("-"*len(title))
        print()

        for issue in items:
            msg = "- `#{0} <{1}>`__: {2}"
            title = re.sub(r"\s+", " ", issue.title.strip())
            if len(title) > 60:
                remainder = re.sub(r"\s.*$", "...", title[60:])
                if len(remainder) > 20:
                    remainder = title[:80] + "..."
                else:
                    title = title[:60] + remainder
            msg = msg.format(issue.id, issue.url, title)
            print(msg)
        print()

    msg = f"Issues closed for {args.milestone}"
    print_list(msg, issues)

    msg = f"Pull requests for {args.milestone}"
    print_list(msg, prs)

    return 0


def get_milestones(getter, project):
    url = f"https://api.github.com/repos/{project}/milestones"
    raw_data, info = getter.get(url)
    data = json.loads(raw_data)

    milestones = {}
    for ms in data:
        milestones[ms['title']] = ms['number']
    return milestones


def get_issues(getter, project, milestone):
    milestones = get_milestones(getter, project)
    mid = milestones[milestone]

    url = "https://api.github.com/repos/{project}/issues?milestone={mid}&state=closed&sort=created&direction=asc"  # noqa: RUF027
    url = url.format(project=project, mid=mid)

    raw_datas = []
    while True:
        raw_data, info = getter.get(url)
        raw_datas.append(raw_data)
        if 'link' not in info:
            break
        m = re.search(r'<(.*?)>; rel="next"', info['link'])
        if m:
            url = m.group(1)
            continue
        break

    issues = [
        Issue(issue_data['number'],
              issue_data['title'],
              issue_data['html_url'])
        for raw_data in raw_datas
        for issue_data in json.loads(raw_data)
    ]
    return issues


class CachedGet:
    def __init__(self, filename):
        self.filename = filename
        if os.path.isfile(filename):
            print(f"[gh_lists] using {filename} as cache (remove it if you want fresh data)",
                  file=sys.stderr)
            with open(filename, 'rb') as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

    def get(self, url):
        if url not in self.cache:
            print("[gh_lists] get:", url, file=sys.stderr)
            req = urlopen(url)
            if req.getcode() != 200:
                raise RuntimeError()
            data = req.read()
            info = dict(req.info())
            self.cache[url] = (data, info)
            req.close()
        else:
            print("[gh_lists] get (cached):", url, file=sys.stderr)
        return self.cache[url]

    def save(self):
        tmp = self.filename + ".new"
        with open(tmp, 'wb') as f:
            json.dump(self.cache, f)
        os.rename(tmp, self.filename)


if __name__ == "__main__":
    sys.exit(main())
