(define (problem t-5)
(:domain blocks)
(:objects a b c d e)
(:init (on e a) (on a d) (ontable d) (clear e)
       (on c b) (ontable b) (clear c)
       (handempty))
(:goal (and (on a b)))
)
