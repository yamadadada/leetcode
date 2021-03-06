package solution600;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Solution697 {

    /**
     * 697. 数组的度
     * @param nums
     * @return
     */
    public int findShortestSubArray(int[] nums) {
        int maxSize = 0;
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(nums[i])) {
                map.get(nums[i]).add(i);
            } else {
                List<Integer> list = new ArrayList<>();
                list.add(i);
                map.put(nums[i], list);
            }
            maxSize = Math.max(maxSize, map.get(nums[i]).size());
        }
        int minAns = Integer.MAX_VALUE;
        for (int key : map.keySet()) {
            if (map.get(key).size() == maxSize) {
                List<Integer> list = map.get(key);
                maxSize = list.size();
                minAns = Math.min(minAns, list.get(list.size() - 1) - list.get(0) + 1);
            }
        }
        return minAns;
    }

    public static void main(String[] args) {
        Solution697 solution = new Solution697();
        int[] nums = {1, 2, 2, 3, 1};
        int ans = solution.findShortestSubArray(nums);
        System.out.println(ans);
    }
}
