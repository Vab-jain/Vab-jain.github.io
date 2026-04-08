---
permalink: /
title: 
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---


I am a Master’s student in Data Science and Artificial Intelligence at [Saarland University](https://www.uni-saarland.de/start.html). Previously, I was a part of the Smart Service Engineering ([SSE](https://www.dfki.de/web/forschung/forschungsbereiche/smart-service-engineering)) group at DFKI Saarbrücken, where I contributed to energy-efficient AI research within the [ESCADE](https://escade-project.de) project.

My master's thesis was on **LLM-Guided Reinforcement Learning in Sparse-Reward Environments**, investigating how schema-constrained LLM guidance can improve exploration in sparse-reward environments. My research interests are in Reinforcement Learning, Agentic Planning, and Foundation Models for RL.

I intend to graduate in early 2026 and am actively seeking opportunities in **RL, LLM-guided planning research, and applied AI/ML roles**.



## Projects
<ul>
{% for project in site.data.projects %}
  <li>
    <strong><a href="{{ project.link }}">{{ project.title }}</a></strong><br>
    {{ project.description }}
  </li>
{% endfor %}
</ul>

## Publications
<ul>
{% for pub in site.data.publications %}
  <li>
    <strong><a href="{{ pub.link }}">{{ pub.title }}</a></strong> ({{ pub.type }})<br>
    {{ pub.authors | markdownify | remove: '<p>' | remove: '</p>' }}<br>
    <em>{{ pub.venue }}</em>, {{ pub.year }}
  </li>
{% endfor %}
</ul>
