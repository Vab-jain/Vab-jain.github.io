---
permalink: /
title: Vaibhav Jain \| AI, Research & More
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

Hi, my name is Vaibhav Jain. I am a Master’s student at [Saarland University](https://www.uni-saarland.de/start.html), enrolled in the Data Science and Artificial Intelligence ([DSAI](https://www.uni-saarland.de/en/study/programmes/master/data-science.html)) program. Since January 2023, I have also been working as a student researcher in the Smart Service Engineering ([SSE](https://www.dfki.de/web/forschung/forschungsbereiche/smart-service-engineering)) group at DFKI Saarbrücken. My master's thesis explores LLM-Guided Reinforcement Learning in Sparse-Reward Environments. I additionally contribute to energy-efficient AI research within the ESCADE project at DFKI. My broadly defined research interests span Knowledge Representation & Reasoning, Agentic Planning, and Foundation Models for RL.

I completed my Bachelor’s degree at the Cluster Innovation Center, University of Delhi. For my Bachelor’s thesis, I worked with [Dr. Santiago Mazuelas](https://www.bcamath.org/en/people/bcam-members/smazuelas) at BCAM in Bilbao, Spain, where I evaluated the robustness of a 0-1 loss min-max classifier under covariate shift.

I am a curious person who enjoys continuously learning and exploring new interests. I love discussing a wide range of topics and always appreciate a good conversation. My hobbies include cooking, going to the gym, playing piano and volleyball, and reading books. I have recently started exploring philosophy and welcome any suggestions!

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
