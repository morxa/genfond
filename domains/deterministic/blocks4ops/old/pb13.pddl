(define (problem pb13)
   (:domain blocksworld)
   (:objects a b c d e f g h i j k l m)
   (:init (ontable a) (ontable b) (ontable c) (ontable d) (ontable e) 
          (ontable f) (ontable g) (ontable h) (ontable i) (ontable j)
          (ontable k)  (ontable l)  (ontable m)
          (clear a)  (clear b) (clear c) (clear d) (clear e) (clear j) 
          (clear f)  (clear g) (clear h) (clear i) (clear k) (clear l) 
          (clear m) (handempty))
   (:goal (and (on a b) (on b c) (on c d) (on d e) (on e f) (on f g)
               (on g h) (on h i) (on i j) (on j k) (on k l)(on l m)  )))


