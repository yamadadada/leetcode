package solution300;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Solution354 {

    /**
     * 354. 俄罗斯套娃信封问题
     * @param envelopes
     * @return
     */
    public int maxEnvelopes(int[][] envelopes) {
        if (envelopes.length == 0) {
            return 0;
        }

        Arrays.sort(envelopes, (e1, e2) -> {
            if (e1[0] == e2[0]) {
                return e2[1] - e1[1];
            } else {
                return e1[0] - e2[0];
            }
        });

        List<Integer> f = new ArrayList<>();
        f.add(envelopes[0][1]);
        for (int i = 1; i < envelopes.length; i++) {
            int num = envelopes[i][1];
            if (num > f.get(f.size() - 1)) {
                f.add(num);
            } else {
                int index = binarySearch(f, num);
                f.set(index, num);
            }
        }
        return f.size();
    }

    /**
     * 二分查找
     * @param list
     * @param target
     * @return
     */
    private int binarySearch(List<Integer> list, int target) {
        int left = 0;
        int right = list.size() - 1;
        while (left < right) {
            int mid = (right - left) / 2 + left;
            if (target > list.get(mid)) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }
}
