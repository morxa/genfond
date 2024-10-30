(define (problem p-13-0)
(:domain blocks)
(:objects l h e a j c d f g k m i b )
(:init (clear b) (clear i) (clear m) (ontable k) (ontable g) (ontable m)
 (on b f) (on f d) (on d c) (on c j) (on j a) (on a e) (on e h) (on h l)
 (on l k) (on i g) (handempty))
(:goal (and (on a b)))

)