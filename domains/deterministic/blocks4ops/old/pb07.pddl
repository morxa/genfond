(define (problem pb7)
   (:domain blocksworld)
   (:objects a b c d e f g)
   (:init (ontable a) (ontable b) (ontable c) (ontable d) (ontable e) 
          (ontable f) (ontable g)
          (clear a)  (clear b) (clear c) (clear d) (clear e) 
          (clear f)  (clear g) (handempty))
   (:goal (and (on a b) (on b c) (on c d) (on d e) (on e f) (on f g))))

