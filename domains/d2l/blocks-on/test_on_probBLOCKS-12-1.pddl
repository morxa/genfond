(define (problem blocks-12-1)
(:domain blocks)
(:objects e l a b f i h g d j k c )
(:init (clear c) (clear k) (ontable j) (ontable d) (on c g) (on g h) (on h i)
 (on i f) (on f b) (on b a) (on a l) (on l e) (on e j) (on k d) (handempty))
(:goal (and (on a b)))

)