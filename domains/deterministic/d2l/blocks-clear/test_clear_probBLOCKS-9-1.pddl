(define (problem blocks-9-1)
(:domain blocks)
(:objects h g i c d b e a f )
(:init (clear f) (ontable a) (on f e) (on e b) (on b d) (on d c) (on c i)
 (on i g) (on g h) (on h a) (handempty))
(:goal (and (clear a)))

)