(define (problem p-7-0)
(:domain blocks)
(:objects c f a b g d e )
(:init (clear e) (ontable d) (on e g) (on g b) (on b a) (on a f) (on f c)
 (on c d) (handempty))
(:goal (and (on a b)))
)