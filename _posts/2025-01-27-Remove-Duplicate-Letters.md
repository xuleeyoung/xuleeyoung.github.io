---
title: 316. Remove Duplicate Letters
date: 2025-01-27
categories: [Tech, LeetCode]
tags: [leetcode]     # TAG names should always be lowercase
---

## Question

Given a string `s`, remove duplicate letters so that every letter appears once and only once. You must make sure your result is **the smallest in lexicographical order** among all possible results.

[316. Remove Duplicate Letters](https://leetcode.com/problems/remove-duplicate-letters/description/)

## Solution

```python
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        stack = []
        letter_counts = dict()
        for ch in s:
            if ch in letter_counts:
                letter_counts[ch] += 1
            else:
                letter_counts[ch] = 1

        for ch in s:
            if ch not in stack:
                while stack and ch < stack[-1] and stack[-1] in letter_counts and letter_counts[stack[-1]] > 0:
                    stack.pop()
                stack.append(ch)
            letter_counts[ch] -= 1

        return ''.join(stack)

```

