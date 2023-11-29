(define (problem pb4)
   (:domain blocksworld)
   (:objects a b c d)
   (:init  (ontable a) (on b a) (on c b) (on d c) (clear d) 
            (handempty))

   (:goal (and (on b a) (on c b) (on a d))))