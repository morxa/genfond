(define (problem problem-1-5)
    (:domain miconic)
    (:objects f0 f1 - floor p0 - passenger)
    (:init (above f0 f1) (destin p0 f1) (lift-at f0) (origin p0 f0))
    (:goal (served p0))
)