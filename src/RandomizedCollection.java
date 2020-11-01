import java.util.*;

/**
 * 381. O(1) 时间插入、删除和获取随机元素 - 允许重复
 */
public class RandomizedCollection {

    private List<Integer> list;

    private Map<Integer, List<Integer>> map;

    /** Initialize your data structure here. */
    public RandomizedCollection() {
        list = new ArrayList<>();
        map = new HashMap<>();
    }

    /** Inserts a value to the collection. Returns true if the collection did not already contain the specified element. */
    public boolean insert(int val) {
        boolean flag = false;
        list.add(val);
        List<Integer> indexList = map.get(val);
        if (indexList == null) {
            indexList = new ArrayList<>();
            flag = true;
        }
        indexList.add(list.size() - 1);
        map.put(val, indexList);
        return flag;
    }

    /** Removes a value from the collection. Returns true if the collection contained the specified element. */
    public boolean remove(int val) {
        List<Integer> indexList = map.get(val);
        if (indexList == null || indexList.size() == 0) {
            return false;
        }
        int deleteIndex = indexList.get(indexList.size() - 1);
        indexList.remove(indexList.size() - 1);
        list.set(deleteIndex, list.get(list.size() - 1));
        list.remove(list.size() - 1);
        return true;
    }

    /** Get a random element from the collection. */
    public int getRandom() {
        return list.get((int) (Math.random() * list.size()));
    }
}

/**
 * Your RandomizedCollection object will be instantiated and called as such:
 * RandomizedCollection obj = new RandomizedCollection();
 * boolean param_1 = obj.insert(val);
 * boolean param_2 = obj.remove(val);
 * int param_3 = obj.getRandom();
 */