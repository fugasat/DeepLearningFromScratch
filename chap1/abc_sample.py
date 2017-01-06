from abc import ABCMeta, abstractmethod


# 抽象クラス
# Abstractなのでインスタンス化できない
class Animal(metaclass=ABCMeta):

    # sub classでoverrideしないとRuntimeError
    @abstractmethod
    def sound(self):
        pass


# 抽象クラスを継承
class Cat(Animal):

    # overrideしないとRuntimeErrorになる
    def sound(self):
        print("Meow")

if __name__ == "__main__":
    #animal = Animal() # Abstractなので初期化できない
    #animal.sound()
    cat = Cat()
    cat.sound()
    print("is animal? : " + str(isinstance(cat, Animal)))
    print("is cat? : " + str(isinstance(cat, Cat)))
