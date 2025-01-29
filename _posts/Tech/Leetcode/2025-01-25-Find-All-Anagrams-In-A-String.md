---
title: 438. Find All Anagrams in a String
date: 2025-01-25
categories: [Tech, LeetCode]
tags: [leetcode]     # TAG names should always be lowercase
---

## Question

Given two strings `s` and `p`, return an array of all the start indices of `p`'s anagrams in `s`. You may return the answer in **any order**.

[438. Find All Anagrams in a String](https://leetcode.com/problems/find-all-anagrams-in-a-string/description/)

## Solution

Sliding Window, Hash table

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        p_dict = {}
        s_dict = {}
        left = right = 0
        for ch in p:
            if ch in p_dict:
                p_dict[ch] += 1
            else:
                p_dict[ch] = 1

        cnt = 0
        ans = []
        while right < len(s):
            if s[right] in p_dict:
                if s[right] in s_dict:
                    s_dict[s[right]] += 1
                else:
                    s_dict[s[right]] = 1
                if s_dict[s[right]] == p_dict[s[right]]:
                    cnt += 1
            
            if right - left + 1 >= len(p):
                if cnt == len(p_dict):
                    ans.append(left)
                if s[left] in p_dict:
                    if s_dict[s[left]] == p_dict[s[left]]:
                        cnt -= 1
                    s_dict[s[left]] -= 1
                left += 1
            
            right += 1
        return ans

```

