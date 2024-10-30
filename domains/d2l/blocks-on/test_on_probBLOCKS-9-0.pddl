(define (problem p-9-0)
(:domain blocks)
(:objects h d i a e g b f c )
(:init (clear c) (clear f) (ontable c) (ontable b) (on f g) (on g e) (on e a)
 (on a i) (on i d) (on d h) (on h b) (handempty))
(:goal (and (on a b)))

)