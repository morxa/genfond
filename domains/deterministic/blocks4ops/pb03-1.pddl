(define (problem pb3-1)
   (:domain blocksworld)
   (:objects a b c)
   (:init (ontable b)   (ontable c)  
          (clear a) (clear b) (on a c) (handempty))
   (:goal (and (on a b) (on b c))))

