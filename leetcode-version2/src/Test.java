import java.util.concurrent.ConcurrentHashMap;

public class Test {
    public static void main(String[] args) {
        ConcurrentHashMap map = new ConcurrentHashMap();
        map.size();
    }
}

class Man {
    int age = 0;
    public void say(){
        System.out.println("man");
    }
}

class Child extends  Man{
    int age = 1;

    public void say(){
        System.out.println("child");
    }

}

class Task implements Runnable{

    @Override
    public void run() {
    }
}