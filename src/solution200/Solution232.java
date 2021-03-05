package solution200;

import java.util.Stack;

public class Solution232 {

    static class MyQueue {

        private final Stack<Integer> inputStack;

        private final Stack<Integer> outputStack;

        /** Initialize your data structure here. */
        public MyQueue() {
            this.inputStack = new Stack<>();
            this.outputStack = new Stack<>();
        }

        /** Push element x to the back of queue. */
        public void push(int x) {
            inputStack.push(x);
        }

        /** Removes the element from in front of queue and returns that element. */
        public int pop() {
            if (outputStack.empty()) {
                while (!inputStack.empty()) {
                    int x = inputStack.pop();
                    outputStack.push(x);
                }
            }
            return outputStack.pop();
        }

        /** Get the front element. */
        public int peek() {
            if (outputStack.empty()) {
                while (!inputStack.empty()) {
                    int x = inputStack.pop();
                    outputStack.push(x);
                }
            }
            return outputStack.peek();
        }

        /** Returns whether the queue is empty. */
        public boolean empty() {
            return inputStack.empty() && outputStack.empty();
        }
    }
}
