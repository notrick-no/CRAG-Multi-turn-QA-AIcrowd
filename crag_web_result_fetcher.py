"""DO NOT MODIFY THIS FILE.

This file has simple functionality to get the full page content of a web search result. During evaluation, `page_content` attribute would already exist. It is advised to use this helper class to fetch the full page content instead of using `requests` directly in your code. Internet access is disabled during evaluation and using `requests` would fail the submission.
根据您提供的代码 `crag_web_result_fetcher.py` 及其顶部的注释，这个脚本在项目中的作用非常明确和关键：

**它是一个用于获取网页搜索结果全文内容的辅助工具，并强制使用缓存机制，以确保代码在本地开发和最终评测两种环境下都能正常运行。**

具体来说，它的功能和重要性体现在以下几点：

1.  **获取网页内容**: 它的核心功能是根据一个URL（`page_url`）来获取整个网页的HTML文本内容（`page_content`）。

2.  **强制缓存 (Caching)**:
    *   当第一次请求某个网页时，它会使用 `requests` 库从互联网下载内容。
    *   下载后，它会将网页内容以文件的形式保存在本地的一个缓存目录中 (`~/.cache/crag/web_search_results`)。
    *   当再次请求同一个网址时，它会直接从本地缓存读取内容，而不会再次发起网络请求，这样可以大大提高效率并节省网络资源。

3.  **兼容评测环境**: 这是该脚本**最重要**的作用。代码顶部的注释明确指出：
    *   **评测期间，服务器会禁用互联网访问**。
    *   直接在你的代码中使用 `requests.get()` 会导致网络请求失败，从而使你的提交（submission）失败。
    *   在评测时，数据集里会预先提供 `page_content` 字段。
    *   这个 `WebSearchResult` 类封装了检查逻辑：它会首先尝试直接获取 `page_content`，如果获取不到（比如在本地开发时），它才会去尝试从缓存或网络获取。

**总结一下：**

这个脚本是一个**强制性的辅助工具**。你（作为参赛者）在开发需要访问网页内容的功能时，**必须**使用这个类，而**不能**直接调用 `requests`。

*   **在本地开发时**，这个类会帮你下载并缓存网页，让你的代码能正常运行。
*   **在最终评测时**，同样的代码不需要任何修改，它会自动切换到使用预先提供的数据，从而绕过被禁用的网络，保证你的提交能成功运行。

简单来说，它为你提供了一个统一、安全的接口来访问网页数据，避免了因环境不同而导致代码失败的问题。
"""

from hashlib import sha256
import os

import requests


CACHE_DIR = os.getenv(
    "CRAG_WEBSEARCH_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache/crag/", "web_search_results"),
)#获取缓存目录，如果缓存目录不存在，则创建
os.makedirs(CACHE_DIR, exist_ok=True)


class WebSearchResult:
    def __init__(self, result: dict):
        self.result = result

    def _get_cache_filename(self) -> str:
        return os.path.join(CACHE_DIR, sha256(self.result["page_url"].encode()).hexdigest())
#首先获取pageurl,.encode()将字符串转化为字节序列 然后计算哈希值，然后拼接缓存目录，返回缓存文件名
    def _page_cache_exists(self) -> bool:
        if os.path.exists(self._get_cache_filename()):
            return True
        return False

    def _fetch_page_content(self) -> str:
        response = requests.get(self.result["page_url"])#requests.get()发送HTTP GET请求，获取网页内容
        response.raise_for_status()
        content = response.text
        with open(self._get_cache_filename(), "w", encoding="utf-8") as f:
            f.write(content)
        self.result["page_content"] = content
        return content

    def _get_page_content_from_cache(self) -> str:
        with open(self._get_cache_filename(), "r", encoding="utf-8") as f:
            return f.read()

    def get(self, key: str, default: str = None) -> str:
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __getitem__(self, key: str) -> str:
        if key == "page_content":
            if "page_content" in self.result:
                return self.result["page_content"]

            if self._page_cache_exists():
                return self._get_page_content_from_cache()
            else:
                return self._fetch_page_content()
        return self.result[key]

    def __len__(self) -> int:
        return len(self.result)

    def __iter__(self):
        return iter(self.result)

    def __getattr__(self, key: str) -> str:
        return self.__getitem__(key)

    def __repr__(self) -> str:
        return str(self.result)

    def __str__(self) -> str:
        return str(self.result)
