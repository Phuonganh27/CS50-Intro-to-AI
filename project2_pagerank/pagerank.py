import os
import random
import re
import sys
import numpy as np

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    probability_dist = {}
    for link in corpus.keys():
        probability_dist[link] = (1-damping_factor)/len(corpus)
        if link in corpus[page]:
            probability_dist[link] += damping_factor/len(corpus[page])
    return probability_dist 
    raise NotImplementedError


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    start_page = random.choice(list(corpus.keys()))
    page_rank = {page: 0 for page in corpus.keys()}
    for i in range(n):
        next_page_prrobability_dist = transition_model(corpus, start_page, damping_factor)
        next_page = random.choices(list(next_page_prrobability_dist.keys()), weights=list(next_page_prrobability_dist.values()), k=1)[0]
        page_rank[next_page] += 1
        start_page = next_page
    for page in page_rank:
        page_rank[page] /= n
    return page_rank
    raise NotImplementedError


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # formula: PR(p) = (1-d)/N + all d PR(i)/Num(i)

    page_list = list(corpus.keys())

    # initialize the page rank, where every page gets equal probability
    page_rank = {page: 1/len(corpus) for page in page_list}

    # find out the pages that have no links at all, if so then link it to all pages
    for source_page, target_pages in corpus.items():
        if len(target_pages) == 0:      
            corpus[source_page] = set(page_list)

    # find out the source pages for each page in the corpus
    sources = {page:set() for page in corpus.keys()}
    for source_page, target_pages in corpus.items():
        for target_page in target_pages:
            if source_page not in sources[target_page]:
                sources[target_page].add(source_page)
    
    # find out the number of links in each page
    numlinks = {page: len(corpus[page]) for page in page_list}

    # iterate until the page rank converges
    # converges when the difference between the new page rank and the old page rank is less than 0.001
    isConvergent = False
    while isConvergent == False:
        new_page_rank = {page: (1-damping_factor)/len(corpus) for page in page_list}
        for target_page in page_list:
            for source_page in sources[target_page]:
                new_page_rank[target_page] += damping_factor*page_rank[source_page]/numlinks[source_page]
                print(new_page_rank)
        new_page_rank_array = np.array(list(new_page_rank.values()))
        page_rank_array = np.array(list(page_rank.values()))

        if np.all(np.abs(new_page_rank_array - page_rank_array) < 0.001):
            isConvergent = True
        else:
            page_rank = new_page_rank
    
    return page_rank
    
    raise NotImplementedError

if __name__ == "__main__":
    main()
