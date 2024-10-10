(define (problem blocks-10-2)
(:domain blocks)
(:objects b g e d f h i a c j )
(:init (clear j) (clear c) (ontable a) (ontable c) (on j i) (on i h) (on h f)
 (on f d) (on d e) (on e g) (on g b) (on b a) (handempty))
(:goal (and (clear a)))

)