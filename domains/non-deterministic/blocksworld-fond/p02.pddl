(define (problem p02)
    (:domain blocks-world-domain)
  (:objects A B C)
  (:init (block A) (block B) (block C) (block Table)
	 (on C A) (on A Table) (on B C)
	 (clear B) (clear Table))
  (:goal (on B A))
)