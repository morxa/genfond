(define (problem pb3-3)
   (:domain blocksworld)
   (:objects a b c)
   (:init (ontable a) (clear c)
          (on c b) (on b a) (handempty))
   (:goal (and (on a b) (on b c))))

