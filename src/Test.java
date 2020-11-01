import common.ListNode;
import common.TreeNode;
import utils.ArrayUtil;
import utils.ListNodeUtil;
import utils.ListUtil;

import java.util.ArrayList;
import java.util.List;

public class Test {

    public static void main(String[] args) {
        Solution solution = new Solution();
        String s = "catsanddog";
        List<String> wordDict = new ArrayList<>();
        wordDict.add("cat");
        wordDict.add("cats");
        wordDict.add("and");
        wordDict.add("sand");
        wordDict.add("dog");
        solution.wordBreak(s, wordDict);
    }
}
