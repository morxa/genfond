(define (problem blocks-6-0)
(:domain blocks)
(:objects e a b c f d )
(:init (clear d) (clear f) (ontable c) (ontable b) (on d a) (on a c) (on f e)
 (on e b) (handempty))
(:goal (and (on a b)))
)