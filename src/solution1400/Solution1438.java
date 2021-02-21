package solution1400;

import java.util.Deque;
import java.util.LinkedList;

public class Solution1438 {

    /**
     * 1438. 绝对差不超过限制的最长连续子数组
     * @param nums
     * @param limit
     * @return
     */
    public int longestSubarray(int[] nums, int limit) {
        Deque<Integer> maxQueue = new LinkedList<>();
        Deque<Integer> minQueue = new LinkedList<>();
        int ans = 0;
        int left = 0;
        int right = 0;
        while (right < nums.length) {
            while (!maxQueue.isEmpty() && maxQueue.peekLast() < nums[right]) {
                maxQueue.pollLast();
            }
            maxQueue.offerLast(nums[right]);
            while (!minQueue.isEmpty() && minQueue.peekLast() > nums[right]) {
                minQueue.pollLast();
            }
            minQueue.offerLast(nums[right]);
            while (!maxQueue.isEmpty() && !minQueue.isEmpty() && maxQueue.peekFirst() - minQueue.peekFirst() > limit) {
                if (nums[left] == maxQueue.peekFirst()) {
                    maxQueue.pollFirst();
                }
                if (nums[left] == minQueue.peekFirst()) {
                    minQueue.pollFirst();
                }
                left++;
            }
            ans = Math.max(ans, right - left + 1);
            right++;
        }
        return ans;
    }

    public static void main(String[] args) {
        int[] nums = {8, 2, 4, 7};
        int limit = 4;
        Solution1438 solution = new Solution1438();
        System.out.println(solution.longestSubarray(nums, limit));
    }
}
