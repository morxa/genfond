(define (problem p-8-1)
(:domain blocks)
(:objects b a g c f d h e )
(:init (clear e) (clear h) (clear d) (clear f) (ontable c) (ontable g)
 (ontable d) (ontable f) (on e c) (on h a) (on a b) (on b g) (handempty))
(:goal (and (on a b)))
)