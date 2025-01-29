---
title: 424. Longest Repeating Character Replacement
date: 2025-01-26
categories: [Tech, LeetCode]
tags: [leetcode]     # TAG names should always be lowercase
---

## Question

You are given a string `s` and an integer `k`. You can choose any character of the string and change it to any other uppercase English character. You can perform this operation at most `k` times.

Return *the length of the longest substring containing the same letter you can get after performing the above operations*.

[424. Longest Repeating Characters Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/)

## Solution

Sliding Window, Hash table

```python
class Solution(object):
    def characterReplacement(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        left = right = 0
        counts = {}
        max_count = 0

        while right < len(s):
            if s[right] in counts:
                counts[s[right]] += 1
            else:
                counts[s[right]] = 1
            
            max_count = max(max_count, counts[s[right]])
            right += 1
            if right - left > k + max_count:
                counts[s[left]] -= 1
                left += 1
            
        return right - left
```

