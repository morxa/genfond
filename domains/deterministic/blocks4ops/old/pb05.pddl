(define (problem pb5)
   (:domain blocksworld)
   (:objects a b c d e)
   (:init (ontable a) (ontable b) (ontable c) (ontable d) (ontable e)  
          (clear a)  (clear b) (clear c) (clear d) (clear e) (handempty))
   (:goal (and (on a b) (on b c) (on c d) (on d e))))


